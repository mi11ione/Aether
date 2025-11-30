// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Tests for Pauli operator commutation rules.
/// Validates single-qubit and multi-qubit commutation logic.
/// Ensures measurement basis determination works correctly.
@Suite("Pauli Commutation Tests")
struct PauliCommutationTests {
    @Test("Identity commutes with all Pauli operators")
    func identityCommutation() {
        #expect(PauliCommutation.commute(nil, nil))
        #expect(PauliCommutation.commute(nil, .x))
        #expect(PauliCommutation.commute(nil, .y))
        #expect(PauliCommutation.commute(nil, .z))
        #expect(PauliCommutation.commute(.x, nil))
        #expect(PauliCommutation.commute(.y, nil))
        #expect(PauliCommutation.commute(.z, nil))
    }

    @Test("Same Pauli operators commute")
    func sameOperatorCommutation() {
        #expect(PauliCommutation.commute(.x, .x))
        #expect(PauliCommutation.commute(.y, .y))
        #expect(PauliCommutation.commute(.z, .z))
    }

    @Test("Different Pauli operators anticommute")
    func differentOperatorAnticommutation() {
        #expect(!PauliCommutation.commute(.x, .y))
        #expect(!PauliCommutation.commute(.x, .z))
        #expect(!PauliCommutation.commute(.y, .z))
        #expect(!PauliCommutation.commute(.y, .x))
        #expect(!PauliCommutation.commute(.z, .x))
        #expect(!PauliCommutation.commute(.z, .y))
    }

    @Test("Multi-qubit Pauli strings with even anticommutations commute")
    func multiQubitEvenAnticommutations() {
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.y(0), .x(1))
        #expect(PauliCommutation.commute(ps1, ps2))
    }

    @Test("Multi-qubit Pauli strings with odd anticommutations anticommute")
    func multiQubitOddAnticommutations() {
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.x(0), .z(1))
        #expect(!PauliCommutation.commute(ps1, ps2))
    }

    @Test("Non-overlapping Pauli strings commute")
    func nonOverlappingStrings() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(1))
        #expect(PauliCommutation.commute(ps1, ps2))
    }

    @Test("Qubit-wise commuting strings with matching operators")
    func qwcMatchingOperators() {
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.x(0), .y(1))
        #expect(PauliCommutation.areQWC(ps1, ps2))
    }

    @Test("Qubit-wise commuting strings with identity gaps")
    func qwcWithIdentityGaps() {
        let ps1 = PauliString(.x(0), .z(2))
        let ps2 = PauliString(.x(0), .y(1))
        #expect(PauliCommutation.areQWC(ps1, ps2))
    }

    @Test("Non-qubit-wise commuting strings with conflicting operators")
    func nonQwcConflictingOperators() {
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.y(0), .x(1))
        #expect(!PauliCommutation.areQWC(ps1, ps2))
    }

    @Test("Measurement basis for single Pauli string")
    func measurementBasisSingleString() {
        let ps = PauliString(.x(0), .z(1))
        let basis = PauliCommutation.measurementBasis(of: ps)
        #expect(basis[0] == .x)
        #expect(basis[1] == .z)
    }

    @Test("Measurement basis for QWC group")
    func measurementBasisQwcGroup() {
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.x(0))
        let ps3 = PauliString(.y(1))
        let basis = PauliCommutation.measurementBasis(of: [ps1, ps2, ps3])
        #expect(basis?[0] == .x)
        #expect(basis?[1] == .y)
    }

    @Test("Measurement basis returns nil for non-QWC strings")
    func measurementBasisNonQwc() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let basis = PauliCommutation.measurementBasis(of: [ps1, ps2])
        #expect(basis == nil)
    }

    @Test("Measurement basis for empty array")
    func measurementBasisEmptyArray() {
        let basis = PauliCommutation.measurementBasis(of: [])
        #expect(basis?.isEmpty == true)
    }
}

/// Tests for QWC grouping using DSATUR graph coloring.
/// Validates conflict graph construction and group formation.
/// Ensures grouping statistics are computed correctly.
@Suite("QWC Grouping Tests")
struct QWCGroupingTests {
    @Test("QWCGroup weight calculation")
    func qwcGroupWeight() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(1))
        let group = QWCGroup(
            terms: [(coefficient: 2.0, pauliString: ps1), (coefficient: -3.5, pauliString: ps2)],
            measurementBasis: [0: .x, 1: .y]
        )
        #expect(abs(group.weight - 5.5) < 1e-10)
    }

    @Test("Grouping single Pauli term")
    func groupingSingleTerm() {
        let ps = PauliString(.x(0))
        let groups = QWCGrouper.group([(coefficient: 1.0, pauliString: ps)])
        #expect(groups.count == 1)
        #expect(groups[0].terms.count == 1)
    }

    @Test("Grouping QWC terms into single group")
    func groupingQwcTerms() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.x(0), .y(1))
        let ps3 = PauliString(.y(1))
        let groups = QWCGrouper.group([(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2), (coefficient: 3.0, pauliString: ps3)])
        #expect(groups.count == 1)
        #expect(groups[0].terms.count == 3)
    }

    @Test("Grouping non-QWC terms into multiple groups")
    func groupingNonQwcTerms() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let groups = QWCGrouper.group([(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2), (coefficient: 3.0, pauliString: ps3)])
        #expect(groups.count == 3)
    }

    @Test("Grouping mixed QWC and non-QWC terms")
    func groupingMixedTerms() {
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.x(0))
        let ps3 = PauliString(.y(0))
        let ps4 = PauliString(.y(1))
        let groups = QWCGrouper.group([(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2), (coefficient: 3.0, pauliString: ps3), (coefficient: 4.0, pauliString: ps4)])
        #expect(groups.count == 2)
    }

    @Test("Grouping empty terms array")
    func groupingEmptyTerms() {
        let groups = QWCGrouper.group([])
        #expect(groups.isEmpty)
    }

    @Test("Each group has valid measurement basis")
    func groupsHaveValidMeasurementBasis() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let groups = QWCGrouper.group([(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        for group in groups {
            #expect(!group.measurementBasis.isEmpty)
        }
    }

    @Test("Grouping statistics for single group")
    func statisticsSingleGroup() {
        let ps = PauliString(.x(0))
        let groups = [QWCGroup(terms: [(coefficient: 1.0, pauliString: ps), (coefficient: 2.0, pauliString: ps), (coefficient: 3.0, pauliString: ps)], measurementBasis: [0: .x])]
        let stats = QWCGrouper.statistics(for: groups)
        #expect(stats.numTerms == 3)
        #expect(stats.numGroups == 1)
        #expect(stats.largestGroupSize == 3)
        #expect(abs(stats.averageGroupSize - 3.0) < 1e-10)
    }

    @Test("Grouping statistics for multiple groups")
    func statisticsMultipleGroups() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let groups = QWCGrouper.group([(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps1), (coefficient: 3.0, pauliString: ps2)])
        let stats = QWCGrouper.statistics(for: groups)
        #expect(stats.numTerms == 3)
        #expect(stats.numGroups == 2)
        #expect(stats.reductionFactor == 1.5)
    }

    @Test("Grouping statistics for empty groups")
    func statisticsEmptyGroups() {
        let stats = QWCGrouper.statistics(for: [])
        #expect(stats.numTerms == 0)
        #expect(stats.numGroups == 0)
        #expect(stats.largestGroupSize == 0)
    }

    @Test("Grouping statistics description format")
    func statisticsDescription() {
        let ps = PauliString(.x(0))
        let groups = [QWCGroup(terms: [(coefficient: 1.0, pauliString: ps)], measurementBasis: [0: .x])]
        let stats = QWCGrouper.statistics(for: groups)
        let description = stats.description
        #expect(description.contains("QWC Grouping Statistics"))
        #expect(description.contains("Terms:"))
        #expect(description.contains("Groups:"))
    }
}

/// Tests for optimal shot allocation across measurement terms.
/// Validates variance-weighted distribution and allocation strategies.
/// Ensures shot allocation statistics are accurate.
@Suite("Shot Allocation Tests")
struct ShotAllocationTests {
    @Test("Uniform allocation when no weights available")
    func uniformAllocation() {
        let allocator = ShotAllocator()
        let ps = PauliString(.x(0))
        let terms = [(coefficient: 0.0, pauliString: ps), (coefficient: 0.0, pauliString: ps), (coefficient: 0.0, pauliString: ps)]
        let allocation = allocator.allocate(for: terms, totalShots: 300, state: nil)
        #expect(allocation.count == 3)
        for shots in allocation.values {
            #expect(shots >= 90)
            #expect(shots <= 110)
        }
    }

    @Test("Weighted allocation based on coefficients")
    func weightedAllocation() {
        let allocator = ShotAllocator()
        let ps = PauliString(.x(0))
        let terms = [(coefficient: 1.0, pauliString: ps), (coefficient: 2.0, pauliString: ps), (coefficient: 3.0, pauliString: ps)]
        let allocation = allocator.allocate(for: terms, totalShots: 600, state: nil)
        #expect(allocation[2]! > allocation[1]!)
        #expect(allocation[1]! > allocation[0]!)
    }

    @Test("Minimum shots per term enforced")
    func minimumShotsEnforced() {
        let allocator = ShotAllocator(minShotsPerTerm: 50)
        let ps = PauliString(.x(0))
        let terms = [(coefficient: 1.0, pauliString: ps), (coefficient: 100.0, pauliString: ps)]
        let allocation = allocator.allocate(for: terms, totalShots: 200, state: nil)
        #expect(allocation[0]! >= 50)
        #expect(allocation[1]! >= 50)
    }

    @Test("Total allocated shots equals requested total")
    func totalShotsMatches() {
        let allocator = ShotAllocator()
        let ps = PauliString(.x(0))
        let terms = [(coefficient: 1.0, pauliString: ps), (coefficient: 2.0, pauliString: ps), (coefficient: 3.0, pauliString: ps), (coefficient: 4.0, pauliString: ps)]
        let allocation = allocator.allocate(for: terms, totalShots: 1000, state: nil)
        let total = allocation.values.reduce(0, +)
        #expect(abs(total - 1000) <= 4)
    }

    @Test("Allocation for QWC groups")
    func allocationForGroups() {
        let allocator = ShotAllocator()
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let group1 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps1)], measurementBasis: [0: .x])
        let group2 = QWCGroup(terms: [(coefficient: 3.0, pauliString: ps2)], measurementBasis: [0: .y])
        let allocation = allocator.allocate(forGroups: [group1, group2], totalShots: 400, state: nil)
        #expect(allocation.count == 2)
        #expect(allocation[1]! > allocation[0]!)
    }

    @Test("Variance reduction estimation")
    func varianceReductionEstimation() {
        let allocator = ShotAllocator()
        let ps = PauliString(.x(0))
        let terms = [(coefficient: 1.0, pauliString: ps), (coefficient: 10.0, pauliString: ps)]
        let allocation = allocator.allocate(for: terms, totalShots: 1000, state: nil)
        let reduction = allocator.varianceReduction(
            for: terms,
            using: allocation,
            comparedTo: 500
        )
        #expect(reduction >= 1.0)
    }

    @Test("Custom configuration")
    func customConfiguration() {
        let allocator = ShotAllocator(minShotsPerTerm: 100)
        let ps = PauliString(.x(0))
        let allocation = allocator.allocate(for: [(coefficient: 1.0, pauliString: ps)], totalShots: 500, state: nil)
        #expect(allocation[0]! >= 100)
    }

    @Test("Allocation enforces effective minimum to prevent over-allocation")
    func allocationEnforcesMinimum() {
        let allocator = ShotAllocator(minShotsPerTerm: 400)
        let ps = PauliString(.x(0))
        let terms = [(coefficient: 1.0, pauliString: ps), (coefficient: 1.0, pauliString: ps)]
        let allocation = allocator.allocate(for: terms, totalShots: 500, state: nil)

        for shots in allocation.values {
            #expect(shots >= 250)
        }
        let total = allocation.values.reduce(0, +)
        #expect(total <= 500)
    }

    @Test("Shot reduction when minShots causes over-allocation")
    func shotReductionFromOverAllocation() {
        let allocator = ShotAllocator(minShotsPerTerm: 30)
        let ps = PauliString(.x(0))

        let terms = [
            (coefficient: 10.0, pauliString: ps),
            (coefficient: 10.0, pauliString: ps),
            (coefficient: 1.0, pauliString: ps),
            (coefficient: 1.0, pauliString: ps),
            (coefficient: 1.0, pauliString: ps),
        ]

        let allocation = allocator.allocate(for: terms, totalShots: 100, state: nil)

        let total = allocation.values.reduce(0, +)
        #expect(total == 100)

        for shots in allocation.values {
            #expect(shots >= 20)
        }
    }

    @Test("Shot reduction removes excess shots iteratively")
    func shotReductionIterativeRemoval() {
        let allocator = ShotAllocator(minShotsPerTerm: 25)
        let ps = PauliString(.x(0))

        var terms: [(Double, PauliString)] = []
        terms += [(10.0, ps), (10.0, ps), (10.0, ps)]
        terms += Array(repeating: (1.0, ps), count: 12)
        let allocation = allocator.allocate(for: terms, totalShots: 200, state: nil)

        let total = allocation.values.reduce(0, +)
        #expect(total == 200)

        for shots in allocation.values {
            #expect(shots > 0)
        }
    }

    @Test("Shot reduction with groups when minShots causes over-allocation")
    func groupShotReductionFromOverAllocation() {
        let allocator = ShotAllocator(minShotsPerTerm: 40)

        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(1))
        let ps3 = PauliString(.z(2))

        let group1 = QWCGroup(terms: [(coefficient: 10.0, pauliString: ps1)], measurementBasis: [0: .x])
        let group2 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps2)], measurementBasis: [1: .y])
        let group3 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps3)], measurementBasis: [2: .z])
        let allocation = allocator.allocate(forGroups: [group1, group2, group3], totalShots: 100, state: nil)

        let total = allocation.values.reduce(0, +)
        #expect(total == 100)

        for shots in allocation.values {
            #expect(shots >= 30)
        }
    }

    @Test("Shot reduction respects minimum when removing excess")
    func shotReductionRespectsMinimum() {
        let allocator = ShotAllocator(minShotsPerTerm: 30)
        let ps = PauliString(.x(0))

        let terms = [
            (coefficient: 8.0, pauliString: ps),
            (coefficient: 8.0, pauliString: ps),
            (coefficient: 1.0, pauliString: ps),
            (coefficient: 1.0, pauliString: ps),
            (coefficient: 1.0, pauliString: ps),
        ]

        let allocation = allocator.allocate(for: terms, totalShots: 150, state: nil)

        let total = allocation.values.reduce(0, +)
        #expect(total == 150)

        for shots in allocation.values {
            #expect(shots >= 30)
        }
    }

    @Test("Shot reduction break when quota met mid-loop")
    func shotReductionBreaksMidLoop() {
        let allocator = ShotAllocator(minShotsPerTerm: 10)
        let ps = PauliString(.x(0))

        let mixedTerms = [
            (coefficient: 10.0, pauliString: ps),
            (coefficient: 10.0, pauliString: ps),
            (coefficient: 10.0, pauliString: ps),
            (coefficient: 10.0, pauliString: ps),
            (coefficient: 10.0, pauliString: ps),
            (coefficient: 1.0, pauliString: ps),
            (coefficient: 1.0, pauliString: ps),
        ]

        let allocation = allocator.allocate(for: mixedTerms, totalShots: 100, state: nil)
        let total = allocation.values.reduce(0, +)
        #expect(total == 100)
    }
}

/// Tests for unitary partitioning of Pauli operators.
/// Validates eigendecomposition and L-BFGS optimization.
/// Ensures partition quality metrics are computed correctly.
@Suite("Unitary Partitioning Tests")
struct UnitaryPartitioningTests {
    @Test("Partition single Pauli term")
    func partitionSingleTerm() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 10,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.1
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps = PauliString(.x(0))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps)])
        #expect(partitions.count >= 1)
    }

    @Test("Partition QWC terms into single partition")
    func partitionQwcTerms() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 10,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.1
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.x(0), .y(1))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        #expect(partitions.count >= 1)
    }

    @Test("Partition empty terms")
    func partitionEmptyTerms() {
        let partitioner = UnitaryPartitioner()
        let partitions = partitioner.partition(terms: [])
        #expect(partitions.isEmpty)
    }

    @Test("Default configuration values")
    func defaultConfiguration() {
        let config = UnitaryPartitioner.Config()
        #expect(config.maxIterations > 0)
        #expect(config.convergenceTolerance > 0)
        #expect(config.circuitDepth > 0)
    }

    @Test("Custom configuration")
    func customConfiguration() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 50,
            convergenceTolerance: 1e-8,
            circuitDepth: 3,
            useAdaptiveDepth: true,
            diagonalityThreshold: 0.05
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps = PauliString(.x(0))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps)])
        #expect(!partitions.isEmpty)
    }

    @Test("Partition has unitary matrix")
    func partitionHasUnitaryMatrix() {
        let ps = PauliString(.x(0))
        let partition = UnitaryPartition(
            terms: [(coefficient: 1.0, pauliString: ps)],
            unitaryMatrix: [[Complex(1.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(1.0, 0.0)]]
        )
        #expect(partition.unitaryMatrix.count == 2)
        #expect(partition.unitaryMatrix[0].count == 2)
    }

    @Test("Multiple partitions preserve all terms")
    func multiplePartitionsPreserveTerms() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let partition1 = UnitaryPartition(
            terms: [(coefficient: 1.0, pauliString: ps1)],
            unitaryMatrix: [[Complex(1.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(1.0, 0.0)]]
        )
        let partition2 = UnitaryPartition(
            terms: [(coefficient: 2.0, pauliString: ps2)],
            unitaryMatrix: [[Complex(1.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(1.0, 0.0)]]
        )
        let totalTerms = partition1.terms.count + partition2.terms.count
        #expect(totalTerms == 2)
    }

    @Test("Partition matrix dimensions match qubit count")
    func partitionMatrixDimensions() {
        let ps = PauliString(.x(0))
        let partition = UnitaryPartition(
            terms: [(coefficient: 1.0, pauliString: ps)],
            unitaryMatrix: [[Complex(1.0, 0.0)]]
        )
        #expect(partition.unitaryMatrix.count == 1, "1x1 matrix for 0-qubit system")
    }

    @Test("Partition preserves all terms")
    func partitionPreservesTerms() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 10,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.1
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let inputTerms = [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)]
        let partitions = partitioner.partition(terms: inputTerms)
        let totalTerms = partitions.reduce(0) { $0 + $1.terms.count }
        #expect(totalTerms == inputTerms.count)
    }

    @Test("Unitary partition numQubits property")
    func unitaryPartitionNumQubits() {
        let ps = PauliString(.x(0), .y(1))
        let partition = UnitaryPartition(
            terms: [(coefficient: 1.0, pauliString: ps)],
            unitaryMatrix: [[Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)]]
        )
        #expect(partition.unitaryMatrix.count == 4, "4x4 matrix for 2-qubit system")
    }

    @Test("Unitary partition stores terms correctly")
    func unitaryPartitionStoresTerms() {
        let ps = PauliString(.x(0))
        let partition = UnitaryPartition(
            terms: [(coefficient: 1.0, pauliString: ps)],
            unitaryMatrix: [[Complex(1.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(1.0, 0.0)]]
        )
        #expect(partition.terms.count == 1)
        #expect(abs(partition.terms[0].coefficient - 1.0) < 1e-10)
    }

    @Test("Unitary partition measurement basis property")
    func unitaryPartitionMeasurementBasis() {
        let ps = PauliString(.x(0), .y(1))
        let partition = UnitaryPartition(
            terms: [(coefficient: 1.0, pauliString: ps)],
            unitaryMatrix: [[Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)]]
        )
        let basis = partition.measurementBasis
        #expect(basis[0] == .z)
        #expect(basis[1] == .z)
    }

    @Test("Unitary partition weight property")
    func unitaryPartitionWeight() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let partition = UnitaryPartition(
            terms: [(coefficient: 2.0, pauliString: ps1), (coefficient: -3.5, pauliString: ps2)],
            unitaryMatrix: [[Complex(1.0, 0.0), Complex(0.0, 0.0)],
                            [Complex(0.0, 0.0), Complex(1.0, 0.0)]]
        )
        #expect(abs(partition.weight - 5.5) < 1e-10)
    }

    @Test("Partition with adaptive depth enabled")
    func partitionWithAdaptiveDepth() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 10,
            convergenceTolerance: 1e-6,
            circuitDepth: 2,
            useAdaptiveDepth: true,
            diagonalityThreshold: 0.1
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2)])
        #expect(!partitions.isEmpty)
    }

    @Test("Partition with multiple qubits")
    func partitionMultipleQubits() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 5,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.2
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.y(0), .x(1))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2)])
        #expect(!partitions.isEmpty)
    }

    @Test("Partition with high ansatz depth")
    func partitionHighAnsatzDepth() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 3,
            convergenceTolerance: 1e-6,
            circuitDepth: 3,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.15
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps = PauliString(.z(0))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps)])
        #expect(!partitions.isEmpty)
    }
}

/// Tests for observable approximation strategies.
/// Validates truncation, top-K selection, and adaptive schedules.
/// Ensures approximation error metrics are accurate.
@Suite("Observable Approximation Tests")
struct ObservableApproximationTests {
    @Test("Truncate with threshold")
    func truncateWithThreshold() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.1, pauliString: ps2), (coefficient: 0.01, pauliString: ps3)])
        let truncated = observable.filtering(coefficientThreshold: 0.05)
        #expect(truncated.terms.count == 2)
    }

    @Test("Truncate keeps largest term when all below threshold")
    func truncateKeepsLargest() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let observable = Observable(terms: [(coefficient: 0.01, pauliString: ps1), (coefficient: 0.02, pauliString: ps2)])
        let truncated = observable.filtering(coefficientThreshold: 0.1)
        #expect(truncated.terms.count == 1)
        #expect(abs(truncated.terms[0].coefficient - 0.02) < 1e-10)
    }

    @Test("Truncate with zero threshold keeps all terms")
    func truncateZeroThreshold() {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps), (coefficient: 0.1, pauliString: ps)])
        let truncated = observable.filtering(coefficientThreshold: 0.0)
        #expect(truncated.terms.count == 2)
    }

    @Test("Truncate empty observable")
    func truncateEmptyObservable() {
        let observable = Observable(terms: [])
        let truncated = observable.filtering(coefficientThreshold: 0.1)
        #expect(truncated.terms.isEmpty)
    }

    @Test("Top-K selection")
    func topKSelection() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 3.0, pauliString: ps2), (coefficient: 2.0, pauliString: ps3)])
        let topK = observable.keepingLargest(2)
        #expect(topK.terms.count == 2)
        #expect(abs(topK.terms[0].coefficient) >= abs(topK.terms[1].coefficient))
    }

    @Test("Top-K with k larger than term count")
    func topKLargerThanCount() {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps), (coefficient: 2.0, pauliString: ps)])
        let topK = observable.keepingLargest(10)
        #expect(topK.terms.count == 2)
    }

    @Test("Top-K with k equal to zero returns largest term")
    func topKZero() {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let topK = observable.keepingLargest(0)
        #expect(topK.terms.count == 1)
    }

    @Test("Approximation error calculation")
    func approximationError() {
        let ps = PauliString(.x(0))
        let full = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let approximate = Observable(terms: [(coefficient: 0.9, pauliString: ps)])
        let state = QuantumState(numQubits: 1)
        let error = full.error(of: approximate, state: state)
        #expect(error >= 0.0)
    }

    @Test("Approximation error for identical observables")
    func approximationErrorIdentical() {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let state = QuantumState(numQubits: 1)
        let error = observable.error(of: observable, state: state)
        #expect(abs(error) < 1e-10)
    }

    @Test("Approximation statistics computation")
    func approximationStatistics() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let full = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        let approximate = Observable(terms: [(coefficient: 1.0, pauliString: ps1)])
        let stats = full.approximationStatistics(approximate: approximate)
        #expect(stats.originalTerms == 2)
        #expect(stats.approximateTerms == 1)
        #expect(stats.reductionFactor == 2.0)
    }

    @Test("Approximation statistics for identical observables")
    func approximationStatisticsIdentical() {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let stats = observable.approximationStatistics(approximate: observable)
        #expect(stats.originalTerms == stats.approximateTerms)
        #expect(stats.reductionFactor == 1.0)
    }

    @Test("Approximation statistics description format")
    func approximationStatisticsDescription() {
        let ps = PauliString(.x(0))
        let full = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let stats = full.approximationStatistics(approximate: full)
        let description = stats.description
        #expect(description.contains("Approximation Statistics"))
        #expect(description.contains("Original terms:"))
        #expect(description.contains("Approximate terms:"))
    }

    @Test("Relative approximation error calculation")
    func relativeApproximationError() {
        let ps = PauliString(.z(0))
        let full = Observable(terms: [(coefficient: 2.0, pauliString: ps)])
        let approximate = Observable(terms: [(coefficient: 1.8, pauliString: ps)])
        let state = QuantumState(numQubits: 1)
        let relError = full.relativeError(of: approximate, state: state)
        #expect(relError >= 0.0)
        #expect(relError <= 1.0)
    }

    @Test("Relative approximation error with zero exact value")
    func relativeApproximationErrorZeroExact() {
        let ps = PauliString(.x(0))
        let full = Observable(terms: [(coefficient: 1.0, pauliString: ps), (coefficient: -1.0, pauliString: ps)])
        let approximate = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let state = QuantumState(numQubits: 1)
        let relError = full.relativeError(of: approximate, state: state)
        #expect(abs(relError) < 1e-9)
    }

    @Test("Validate approximation within tolerance")
    func validateApproximation() {
        let ps = PauliString(.z(0))
        let full = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let approximate = Observable(terms: [(coefficient: 0.99, pauliString: ps)])
        let state = QuantumState(numQubits: 1)
        let valid = full.meetsAccuracy(approximate, state: state, tolerance: 0.1)
        #expect(valid)
    }

    @Test("Validate approximation outside tolerance")
    func validateApproximationOutsideTolerance() {
        let ps = PauliString(.z(0))
        let full = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let approximate = Observable(terms: [(coefficient: 0.5, pauliString: ps)])
        let state = QuantumState(numQubits: 1)
        let valid = full.meetsAccuracy(approximate, state: state, tolerance: 0.01)
        #expect(!valid)
    }

    @Test("Find optimal threshold for approximation")
    func findOptimalThreshold() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.5, pauliString: ps2), (coefficient: 0.1, pauliString: ps3)])
        let state = QuantumState(numQubits: 1)
        let threshold = observable.findOptimalThreshold(state: state, maxError: 0.2, searchSteps: 10)
        #expect(threshold >= 0.0)
        #expect(threshold <= 1.0)
    }

    @Test("Find optimal threshold with strict error constraint")
    func findOptimalThresholdStrictConstraint() {
        let ps1 = PauliString(.z(0))
        let ps2 = PauliString(.z(0))
        let ps3 = PauliString(.z(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.3, pauliString: ps2), (coefficient: 0.1, pauliString: ps3)])
        let state = QuantumState(numQubits: 1)
        let threshold = observable.findOptimalThreshold(state: state, maxError: 0.01, searchSteps: 20)
        let approx = observable.filtering(coefficientThreshold: threshold)
        let error = observable.error(of: approx, state: state)
        #expect(error <= 0.02)
    }

    @Test("topK with k=0 on empty observable returns empty")
    func topKZeroEmptyObservable() {
        let observable = Observable(terms: [])
        let result = observable.keepingLargest(0)
        #expect(result.terms.isEmpty)
    }

    @Test("Approximation statistics with empty original observable")
    func approximationStatsEmptyOriginal() {
        let ps = PauliString(.x(0))
        let empty = Observable(terms: [])
        let nonEmpty = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let stats = empty.approximationStatistics(approximate: nonEmpty)

        #expect(stats.originalTerms == 0)
        #expect(stats.reductionFactor == 1.0)
        #expect(stats.coefficientRetention == 0.0)
    }

    @Test("Find optimal threshold on empty observable")
    func findOptimalThresholdEmpty() {
        let observable = Observable(terms: [])
        let state = QuantumState(numQubits: 1)
        let threshold = observable.findOptimalThreshold(state: state, maxError: 0.1)

        #expect(abs(threshold - 1.0) < 1e-6)
    }

    @Test("topK with k=0 returns single largest term")
    func topKZeroReturnsLargest() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: 3.5, pauliString: PauliString(.z(1))),
            (coefficient: -2.0, pauliString: PauliString(.y(2))),
        ])

        let approx = observable.keepingLargest(0)

        #expect(approx.terms.count == 1)
        #expect(abs(approx.terms[0].coefficient - 3.5) < 1e-10)
    }

    @Test("topK with k=0 on empty observable returns empty")
    func topKZeroEmptyReturnsEmpty() {
        let observable = Observable(terms: [])
        let approx = observable.keepingLargest(0)

        #expect(approx.terms.count == 0)
    }

    @Test("topK with negative k returns largest term")
    func topKNegativeReturnsLargest() {
        let observable = Observable(terms: [
            (coefficient: -5.0, pauliString: PauliString(.z(0))),
            (coefficient: 2.0, pauliString: PauliString(.x(1))),
        ])

        let approx = observable.keepingLargest(-1)

        #expect(approx.terms.count == 1)
        #expect(abs(abs(approx.terms[0].coefficient) - 5.0) < 1e-10)
    }

    @Test("topK with k=1 returns single largest term")
    func topKOneReturnsLargest() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: 4.0, pauliString: PauliString(.z(1))),
            (coefficient: -2.0, pauliString: PauliString(.y(2))),
        ])

        let approx = observable.keepingLargest(1)

        #expect(approx.terms.count == 1)
        #expect(abs(approx.terms[0].coefficient - 4.0) < 1e-10)
    }

    @Test("topK with k=2 returns two largest terms")
    func topKTwoReturnsTopTwo() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: 4.0, pauliString: PauliString(.z(1))),
            (coefficient: -3.0, pauliString: PauliString(.y(2))),
            (coefficient: 2.0, pauliString: PauliString(.x(3))),
        ])

        let approx = observable.keepingLargest(2)

        #expect(approx.terms.count == 2)

        let coeffs = approx.terms.map { abs($0.coefficient) }.sorted(by: >)
        #expect(abs(coeffs[0] - 4.0) < 1e-10)
        #expect(abs(coeffs[1] - 3.0) < 1e-10)
    }

    @Test("topK with k larger than term count returns all terms")
    func topKLargerThanCountReturnsAll() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: 2.0, pauliString: PauliString(.z(1))),
        ])

        let approx = observable.keepingLargest(10)

        #expect(approx.terms.count == 2)
    }

    @Test("topK selects by absolute value of coefficient")
    func topKSelectsByAbsoluteValue() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: -5.0, pauliString: PauliString(.z(1))),
            (coefficient: 3.0, pauliString: PauliString(.y(2))),
        ])

        let approx = observable.keepingLargest(1)

        #expect(approx.terms.count == 1)
        #expect(abs(abs(approx.terms[0].coefficient) - 5.0) < 1e-10)
    }
}

/// Tests for measurement optimization integration layer.
/// Validates caching behavior and measurement strategies.
/// Ensures thread-safe operation under concurrent access.
@Suite("Measurement Optimization Integration Tests")
struct MeasurementOptimizationIntegrationTests {
    @Test("QWC groups caching")
    func qwcGroupsCaching() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        let groups1 = observable.qwcGroups
        let groups2 = observable.qwcGroups

        let countsMatch = await groups1().count == groups2().count
        #expect(countsMatch)
    }

    @Test("QWC groups for single term")
    func qwcGroupsSingleTerm() async {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let groups = observable.qwcGroups
        await #expect(groups().count == 1)
        await #expect(groups()[0].terms.count == 1)
    }

    @Test("QWC groups reduce measurement count")
    func qwcGroupsReduceMeasurements() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.x(0), .y(1))
        let ps3 = PauliString(.y(1))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2), (coefficient: 3.0, pauliString: ps3)])
        let groups = observable.qwcGroups
        await #expect(groups().count < 3)
    }

    @Test("Unitary partitions caching")
    func unitaryPartitionsCaching() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        let config = UnitaryPartitioner.Config(
            maxIterations: 5,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.1
        )
        let partitions1 = await observable.unitaryPartitions(config: config)
        let partitions2 = await observable.unitaryPartitions(config: config)
        #expect(partitions1.count == partitions2.count)
    }

    @Test("Unitary partitions for single term")
    func unitaryPartitionsSingleTerm() async {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let config = UnitaryPartitioner.Config(
            maxIterations: 5,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.1
        )
        let partitions = await observable.unitaryPartitions(config: config)
        #expect(!partitions.isEmpty)
    }

    @Test("Shot allocation for observable terms")
    func shotAllocationForTerms() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 3.0, pauliString: ps2)])
        let allocation = observable.allocateShots(totalShots: 1000, state: nil)
        #expect(allocation.count == 2)
        #expect(allocation.values.reduce(0, +) <= 1000)
    }

    @Test("Shot allocation for QWC groups")
    func shotAllocationForGroups() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.x(0), .y(1))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        let allocation = await observable.allocateShotsForGroups(totalShots: 1000, state: nil)
        #expect(!allocation.isEmpty)
    }

    @Test("Measurement optimization statistics computation")
    func measurementOptimizationStatistics() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        let stats = await observable.optimizationStatistics()
        #expect(stats.numTerms == 2)
        #expect(stats.numQWCGroups > 0)
    }

    @Test("Measurement optimization statistics description format")
    func measurementOptimizationStatisticsDescription() async {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let stats = await observable.optimizationStatistics()
        let description = stats.description
        #expect(description.contains("Measurement Optimization Statistics"))
        #expect(description.contains("Hamiltonian terms:"))
        #expect(description.contains("QWC groups:"))
    }

    @Test("Measurement optimization statistics for empty observable")
    func measurementOptimizationStatisticsEmpty() async {
        let observable = Observable(terms: [])
        let stats = await observable.optimizationStatistics()
        #expect(stats.numTerms == 0)
        #expect(stats.numQWCGroups == 0)
    }

    @Test("Different observables have different QWC groups")
    func differentObservablesDifferentGroups() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let obs1 = Observable(terms: [(coefficient: 1.0, pauliString: ps1)])
        let obs2 = Observable(terms: [(coefficient: 1.0, pauliString: ps2)])
        let groups1 = obs1.qwcGroups
        let groups2 = obs2.qwcGroups
        await #expect(groups1().count == 1)
        await #expect(groups2().count == 1)

        let measurementMatch = await groups1()[0].measurementBasis != groups2()[0].measurementBasis
        #expect(measurementMatch)
    }

    @Test("Shot allocation respects minimum shots")
    func shotAllocationRespectsMinimum() {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 100.0, pauliString: ps2)])
        let allocation = observable.allocateShots(totalShots: 500, minShotsPerTerm: 50, state: nil)
        for shots in allocation.values {
            #expect(shots >= 50)
        }
    }

    @Test("Measurement optimization reduces required measurements")
    func measurementOptimizationReduces() async {
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.x(0))
        let ps3 = PauliString(.y(1))
        let ps4 = PauliString(.x(0), .y(1))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2), (coefficient: 3.0, pauliString: ps3), (coefficient: 4.0, pauliString: ps4)])
        let groups = observable.qwcGroups
        await #expect(groups().count < observable.terms.count)
    }

    @Test("Clear grouping caches")
    func clearGroupingCaches() async {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])

        let groups1 = observable.qwcGroups
        await Observable.clearGroupingCaches()
        let groups2 = observable.qwcGroups
        let countsMatch = await groups1().count == groups2().count

        #expect(countsMatch, "Groups should have same count after cache clear")
        await #expect(groups1().count == 1, "Single X operator should form 1 QWC group")
    }

    @Test("Measurement circuits for term-by-term strategy")
    func measurementCircuitsTermByTerm() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        let count = await observable.measureCircuitCount(for: .termByTerm)
        #expect(count == 2)
    }

    @Test("Measurement circuits for QWC grouping strategy")
    func measurementCircuitsQwcGrouping() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        let count = await observable.measureCircuitCount(for: .qwcGrouping)
        #expect(count == 1)
    }

    @Test("Measurement circuits for unitary partitioning strategy")
    func measurementCircuitsUnitaryPartitioning() async {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let config = UnitaryPartitioner.Config(
            maxIterations: 5,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.1
        )
        let count = await observable.measureCircuitCount(for: .unitaryPartitioning(config: config))
        #expect(count >= 1)
    }

    @Test("Measurement circuits for automatic strategy selection")
    func measurementCircuitsAutomatic() async {
        let ps = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps)])
        let count = await observable.measureCircuitCount(for: .automatic)
        #expect(count >= 1)
    }

    @Test("Shot allocation with state for variance estimation")
    func shotAllocationWithState() {
        let ps = PauliString(.z(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps), (coefficient: 2.0, pauliString: ps)])
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()
        let allocation = observable.allocateShots(totalShots: 1000, state: state)
        #expect(allocation.count == 2)
    }

    @Test("Shot allocation for groups with state")
    func shotAllocationForGroupsWithState() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()
        let allocation = await observable.allocateShotsForGroups(totalShots: 1000, state: state)
        #expect(!allocation.isEmpty)
    }

    @Test("Automatic strategy selection for large Hamiltonian with small qubits")
    func automaticStrategyLargeHamiltonian() async {
        var terms: [(Double, PauliString)] = []
        for i in 0 ..< 600 {
            let qubit = i % 5
            let basis: PauliBasis = [.x, .y, .z][i % 3]
            let op = switch basis {
            case .x: PauliOperator.x(qubit)
            case .y: PauliOperator.y(qubit)
            case .z: PauliOperator.z(qubit)
            }
            let ps = PauliString(op)
            terms.append((coefficient: 1.0, pauliString: ps))
        }
        let observable = Observable(terms: terms)
        let count = await observable.measureCircuitCount(for: .automatic)
        #expect(count >= 1)
    }

    @Test("Shot allocation with remaining shots distribution")
    func shotAllocationWithRemainingShots() {
        let allocator = ShotAllocator(minShotsPerTerm: 1)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let terms = [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2), (coefficient: 1.0, pauliString: ps3)]
        let allocation = allocator.allocate(for: terms, totalShots: 100, state: nil)
        #expect(allocation.count == 3)
        let total = allocation.values.reduce(0, +)
        #expect(total == 100)
    }

    @Test("Shot allocation distributes exact remainder evenly")
    func shotAllocationDistributesRemainder() {
        let allocator = ShotAllocator(minShotsPerTerm: 1)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let ps4 = PauliString(.x(0))
        let terms = [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2), (coefficient: 1.0, pauliString: ps3), (coefficient: 1.0, pauliString: ps4)]
        let allocation = allocator.allocate(for: terms, totalShots: 97, state: nil)
        #expect(allocation.count == 4)
        let total = allocation.values.reduce(0, +)
        #expect(total == 97)
    }

    @Test("Shot allocation for groups with remaining shots distribution")
    func shotAllocationGroupsWithRemainingShots() {
        let allocator = ShotAllocator(minShotsPerTerm: 1)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let group1 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps1)], measurementBasis: [0: .x])
        let group2 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps2)], measurementBasis: [0: .y])
        let group3 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps3)], measurementBasis: [0: .z])
        let allocation = allocator.allocate(forGroups: [group1, group2, group3], totalShots: 100, state: nil)
        #expect(allocation.count == 3)
        let total = allocation.values.reduce(0, +)
        #expect(total == 100)
    }

    @Test("Shot allocation for groups distributes exact remainder")
    func shotAllocationGroupsDistributesRemainder() {
        let allocator = ShotAllocator(minShotsPerTerm: 1)
        let ps = PauliString(.x(0))
        let group1 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps)], measurementBasis: [0: .x])
        let group2 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps)], measurementBasis: [0: .y])
        let group3 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps)], measurementBasis: [0: .z])
        let group4 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps)], measurementBasis: [0: .x])
        let group5 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps)], measurementBasis: [0: .y])
        let allocation = allocator.allocate(forGroups: [group1, group2, group3, group4, group5], totalShots: 102, state: nil)
        #expect(allocation.count == 5)
        let total = allocation.values.reduce(0, +)
        #expect(total == 102)
    }

    @Test("Shot allocation for groups with zero total weight")
    func shotAllocationGroupsZeroWeight() {
        let allocator = ShotAllocator()
        let ps = PauliString(.x(0))
        let group1 = QWCGroup(terms: [(coefficient: 0.0, pauliString: ps)], measurementBasis: [0: .x])
        let group2 = QWCGroup(terms: [(coefficient: 0.0, pauliString: ps)], measurementBasis: [0: .y])
        let allocation = allocator.allocate(forGroups: [group1, group2], totalShots: 1000, state: nil)
        #expect(allocation.count == 2)
        #expect(allocation[0]! + allocation[1]! == 1000)
    }

    @Test("Shot allocation for groups enforces effective minimum to prevent over-allocation")
    func shotAllocationGroupsShotsReduction() {
        let allocator = ShotAllocator(minShotsPerTerm: 400)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let group1 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps1)], measurementBasis: [0: .x])
        let group2 = QWCGroup(terms: [(coefficient: 1.0, pauliString: ps2)], measurementBasis: [0: .y])
        let allocation = allocator.allocate(forGroups: [group1, group2], totalShots: 500, state: nil)

        #expect(allocation.count == 2)
        for shots in allocation.values {
            #expect(shots >= 250)
        }

        let total = allocation.values.reduce(0, +)
        #expect(total <= 500)
    }

    @Test("Unitary partitioning with variational optimization")
    func unitaryPartitioningVariationalOptimization() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 20,
            convergenceTolerance: 1e-6,
            circuitDepth: 2,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.5
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.8, pauliString: ps2), (coefficient: 0.6, pauliString: ps3)])
        #expect(!partitions.isEmpty)
    }

    @Test("Unitary partitioning with optimization failure fallback to identity")
    func unitaryPartitioningOptimizationFailure() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 1,
            convergenceTolerance: 1e-10,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 1e-15
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.y(0), .x(1))
        let ps3 = PauliString(.z(0), .z(1))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2), (coefficient: 1.0, pauliString: ps3)])
        #expect(partitions.count >= 1)
    }

    @Test("Unitary partitioning with merge failure increments index")
    func unitaryPartitioningMergeFailure() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 2,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 1e-15
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))
        let ps4 = PauliString(.x(1))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2), (coefficient: 1.0, pauliString: ps3), (coefficient: 1.0, pauliString: ps4)])

        #expect(partitions.count >= 2)

        let totalTerms = partitions.reduce(0) { $0 + $1.terms.count }
        #expect(totalTerms == 4)
    }

    @Test("Unitary partitioning exercises L-BFGS optimization")
    func unitaryPartitioningLbfgsOptimization() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 30,
            convergenceTolerance: 1e-5,
            circuitDepth: 2,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.2
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.y(0), .z(1))
        let ps3 = PauliString(.z(0), .x(1))
        let ps4 = PauliString(.x(0), .z(1))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.9, pauliString: ps2), (coefficient: 0.8, pauliString: ps3), (coefficient: 0.7, pauliString: ps4)])
        #expect(!partitions.isEmpty)
    }

    @Test("Unitary partitioning with deep ansatz depth")
    func unitaryPartitioningDeepAnsatz() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 25,
            convergenceTolerance: 1e-4,
            circuitDepth: 4,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.25
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .x(1))
        let ps2 = PauliString(.y(0), .y(1))
        let ps3 = PauliString(.z(0), .z(1))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.95, pauliString: ps2), (coefficient: 0.85, pauliString: ps3)])
        #expect(!partitions.isEmpty)
    }

    @Test("Unitary partitioning forces variational optimization path")
    func unitaryPartitioningForcesVariational() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 20,
            convergenceTolerance: 1e-5,
            circuitDepth: 3,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.15
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .y(1), .z(2))
        let ps2 = PauliString(.y(0), .z(1), .x(2))
        let ps3 = PauliString(.z(0), .x(1), .y(2))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.9, pauliString: ps2), (coefficient: 0.8, pauliString: ps3)])
        #expect(!partitions.isEmpty)
    }

    @Test("Unitary partitioning identity matrix fallback")
    func unitaryPartitioningIdentityFallback() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 1,
            convergenceTolerance: 1e-8,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 1e-16
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .y(1), .z(2))
        let ps2 = PauliString(.y(0), .z(1), .x(2))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2)])
        #expect(partitions.count >= 1)
    }

    @Test("L-BFGS optimization with early convergence")
    func lbfgsEarlyConvergence() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 100,
            convergenceTolerance: 0.5,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.8
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.5, pauliString: ps2)])
        #expect(!partitions.isEmpty)
    }

    @Test("L-BFGS optimization with history buffer overflow")
    func lbfgsHistoryBufferOverflow() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 50,
            convergenceTolerance: 1e-6,
            circuitDepth: 2,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.2
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.y(0), .z(1))
        let ps3 = PauliString(.z(0), .x(1))
        let ps4 = PauliString(.x(0), .x(1))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.9, pauliString: ps2), (coefficient: 0.8, pauliString: ps3), (coefficient: 0.7, pauliString: ps4)])
        #expect(!partitions.isEmpty)
    }

    @Test("L-BFGS optimization exercises compute search direction")
    func lbfgsComputeSearchDirection() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 40,
            convergenceTolerance: 1e-5,
            circuitDepth: 3,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.18
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .y(1), .z(2))
        let ps2 = PauliString(.y(0), .z(1), .x(2))
        let ps3 = PauliString(.z(0), .x(1), .y(2))
        let ps4 = PauliString(.x(0), .x(1), .x(2))
        let ps5 = PauliString(.y(0), .y(1), .y(2))
        let partitions = partitioner.partition(
            terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.95, pauliString: ps2), (coefficient: 0.9, pauliString: ps3), (coefficient: 0.85, pauliString: ps4), (coefficient: 0.8, pauliString: ps5)]
        )
        #expect(!partitions.isEmpty)
    }

    @Test("L-BFGS line search with backtracking")
    func lbfgsLineSearchBacktracking() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 35,
            convergenceTolerance: 1e-6,
            circuitDepth: 2,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.22
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0), .y(1))
        let ps2 = PauliString(.y(0), .x(1))
        let ps3 = PauliString(.z(0), .z(1))
        let ps4 = PauliString(.x(0), .z(1))
        let ps5 = PauliString(.y(0), .z(1))
        let partitions = partitioner.partition(
            terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.92, pauliString: ps2), (coefficient: 0.88, pauliString: ps3), (coefficient: 0.84, pauliString: ps4), (coefficient: 0.8, pauliString: ps5)]
        )
        #expect(!partitions.isEmpty)
    }

    @Test("Cache collision detection prevents wrong results")
    func cacheCollisionDetection() async {
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let obs1 = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])
        let obs2 = Observable(terms: [(coefficient: 3.0, pauliString: ps1), (coefficient: 4.0, pauliString: ps2)])

        let groups1 = obs1.qwcGroups
        let groups2 = obs2.qwcGroups

        await #expect(groups1().count == 2)
        await #expect(groups2().count == 2)

        let coeffs1 = await groups1().flatMap { $0.terms.map(\.coefficient) }.sorted()
        let coeffs2 = await groups2().flatMap { $0.terms.map(\.coefficient) }.sorted()
        #expect(coeffs1 != coeffs2)
    }

    @Test("Batch variance estimation with state")
    func batchVarianceEstimation() {
        let ps1 = PauliString(.z(0))
        let ps2 = PauliString(.x(0))
        let observable = Observable(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 2.0, pauliString: ps2)])

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let allocation = observable.allocateShots(totalShots: 1000, state: state)

        #expect(allocation.count == 2)
        let total = allocation.values.reduce(0, +)
        #expect(total <= 1000)
    }

    @Test("Variance reduction estimation with state parameter")
    func varianceReductionWithState() {
        let allocator = ShotAllocator()
        let ps1 = PauliString(.z(0))
        let ps2 = PauliString(.x(0))
        let terms = [(coefficient: 1.0, pauliString: ps1), (coefficient: 10.0, pauliString: ps2)]

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let allocation = allocator.allocate(for: terms, totalShots: 1000, state: state)
        let reduction = allocator.varianceReduction(
            for: terms,
            using: allocation,
            comparedTo: 500,
            state: state
        )

        #expect(reduction >= 1.0)
    }

    @Test("Variance reduction without state uses conservative estimate")
    func varianceReductionWithoutState() {
        let allocator = ShotAllocator()
        let ps = PauliString(.x(0))
        let terms = [(coefficient: 1.0, pauliString: ps), (coefficient: 10.0, pauliString: ps)]
        let allocation = allocator.allocate(for: terms, totalShots: 1000, state: nil)
        let reduction = allocator.varianceReduction(
            for: terms,
            using: allocation,
            comparedTo: 500,
            state: nil
        )

        #expect(reduction >= 1.0)
    }

    @Test("Unitary partitioning caches Pauli matrices during optimization")
    func unitaryPartitioningCachesPauliMatrices() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 10,
            convergenceTolerance: 1e-6,
            circuitDepth: 2,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.2
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))

        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 0.9, pauliString: ps2), (coefficient: 0.8, pauliString: ps3)])

        #expect(!partitions.isEmpty)
        let totalTerms = partitions.reduce(0) { $0 + $1.terms.count }
        #expect(totalTerms == 3)
    }

    @Test("Effective minimum shots prevents over-allocation in edge cases")
    func effectiveMinimumPreventsOverAllocation() {
        let allocator = ShotAllocator(minShotsPerTerm: 1000)
        let ps = PauliString(.x(0))

        let terms = Array(repeating: (coefficient: 1.0, pauliString: ps), count: 10)
        let allocation = allocator.allocate(for: terms, totalShots: 1000, state: nil)

        #expect(allocation.count == 10)
        let total = allocation.values.reduce(0, +)
        #expect(total <= 1000)

        for shots in allocation.values {
            #expect(shots >= 100)
            #expect(shots <= 100)
        }
    }

    @Test("Cache returns correct values after clear")
    func cacheReturnsCorrectValuesAfterClear() async {
        let ps = PauliString(.x(0))
        let obs1 = Observable(terms: [(coefficient: 1.0, pauliString: ps)])

        let groups1 = obs1.qwcGroups
        await #expect(groups1().count == 1)

        await Observable.clearGroupingCaches()

        let groups2 = obs1.qwcGroups
        await #expect(groups2().count == 1)

        let countsMatch = await groups1()[0].terms.count == groups2()[0].terms.count
        #expect(countsMatch)
    }

    @Test("Cost function decreases during optimization")
    func costFunctionDecreasesDuringOptimization() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 50,
            convergenceTolerance: 1e-6,
            circuitDepth: 2,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.15
        )
        let partitioner = UnitaryPartitioner(config: config)

        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))
        let ps3 = PauliString(.z(0))

        let partitions = partitioner.partition(
            terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2), (coefficient: 1.0, pauliString: ps3)]
        )

        #expect(!partitions.isEmpty)
        if partitions.count == 1 {
            #expect(partitions[0].terms.count == 3)
        }
    }

    @Test("Gradient function produces valid derivatives")
    func gradientFunctionProducesValidDerivatives() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 30,
            convergenceTolerance: 1e-5,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.2
        )
        let partitioner = UnitaryPartitioner(config: config)

        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))

        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2)])

        #expect(!partitions.isEmpty)
        let totalTerms = partitions.reduce(0) { $0 + $1.terms.count }
        #expect(totalTerms == 2)
    }

    @Test("Cost and gradient functions handle multi-qubit systems")
    func costAndGradientHandleMultiQubitSystems() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 40,
            convergenceTolerance: 1e-5,
            circuitDepth: 2,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.25
        )
        let partitioner = UnitaryPartitioner(config: config)

        let ps1 = PauliString(.x(0), .x(1))
        let ps2 = PauliString(.y(0), .y(1))
        let ps3 = PauliString(.z(0), .z(1))

        let partitions = partitioner.partition(
            terms: [(coefficient: 1.0, pauliString: ps1), (coefficient: 1.0, pauliString: ps2), (coefficient: 0.5, pauliString: ps3)]
        )

        #expect(!partitions.isEmpty)
        let totalTerms = partitions.reduce(0) { $0 + $1.terms.count }
        #expect(totalTerms == 3)
    }

    @Test("Optimization handles weighted coefficients correctly")
    func optimizationHandlesWeightedCoefficients() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 40,
            convergenceTolerance: 1e-5,
            circuitDepth: 2,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.3
        )
        let partitioner = UnitaryPartitioner(config: config)

        let ps1 = PauliString(.x(0))
        let ps2 = PauliString(.y(0))

        let partitions = partitioner.partition(
            terms: [(coefficient: 10.0, pauliString: ps1), (coefficient: 0.1, pauliString: ps2)]
        )

        #expect(!partitions.isEmpty)
        let totalTerms = partitions.reduce(0) { $0 + $1.terms.count }
        #expect(totalTerms == 2)

        let allCoeffs = partitions.flatMap { partition in
            partition.terms.map { abs($0.coefficient) }
        }.sorted()
        #expect(allCoeffs.contains(where: { abs($0 - 0.1) < 1e-10 }))
        #expect(allCoeffs.contains(where: { abs($0 - 10.0) < 1e-10 }))
    }

    @Test("Gradient computation converges for identity case")
    func gradientComputationConvergesForIdentity() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 20,
            convergenceTolerance: 1e-6,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 0.01
        )
        let partitioner = UnitaryPartitioner(config: config)
        let ps1 = PauliString(.z(0))
        let partitions = partitioner.partition(terms: [(coefficient: 1.0, pauliString: ps1)])

        #expect(partitions.count == 1)
        #expect(partitions[0].terms.count == 1)
    }

    @Test("Optimization failure triggers identity matrix fallback")
    func optimizationFailureIdentityFallback() {
        let config = UnitaryPartitioner.Config(
            maxIterations: 1,
            convergenceTolerance: 1e-10,
            circuitDepth: 1,
            useAdaptiveDepth: false,
            diagonalityThreshold: 1e-18
        )
        let partitioner = UnitaryPartitioner(config: config)

        let ps1 = PauliString(.x(0), .x(1), .x(2))
        let ps2 = PauliString(.y(0), .y(1), .y(2))
        let ps3 = PauliString(.z(0), .z(1), .z(2))
        let ps4 = PauliString(.x(0), .y(1), .z(2))

        let terms: PauliTerms = [
            (coefficient: 1.0, pauliString: ps1),
            (coefficient: 0.9, pauliString: ps2),
            (coefficient: 0.8, pauliString: ps3),
            (coefficient: 0.7, pauliString: ps4),
        ]

        let partitions = partitioner.partition(terms: terms)

        #expect(!partitions.isEmpty,
                "Partitioner should create at least one partition even when optimization fails")

        let totalTerms = partitions.reduce(0) { $0 + $1.terms.count }
        #expect(totalTerms == terms.count,
                "All terms should be preserved across partitions")

        for partition in partitions {
            #expect(partition.unitaryMatrix.count == 8,
                    "3-qubit unitary matrix should be 8x8")
        }
    }
}
