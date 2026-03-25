// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Test suite for DMRGConfiguration struct.
/// Validates default and custom parameter initialization, property access,
/// and Equatable conformance for DMRG algorithm configuration.
@Suite("DMRGConfiguration")
struct DMRGConfigurationTests {
    @Test("Default configuration has expected values")
    func defaultConfigurationValues() {
        let config = DMRGConfiguration()

        #expect(config.maxSweeps == 20, "Default maxSweeps should be 20")
        #expect(abs(config.convergenceThreshold - 1e-8) < 1e-15, "Default convergenceThreshold should be 1e-8")
        #expect(config.isSubspaceExpansionEnabled == false, "Default isSubspaceExpansionEnabled should be false")
        #expect(abs(config.noiseStrength - 0.0) < 1e-15, "Default noiseStrength should be 0.0")
    }

    @Test("Custom maxSweeps configuration")
    func customMaxSweeps() {
        let config = DMRGConfiguration(maxSweeps: 50)

        #expect(config.maxSweeps == 50, "maxSweeps should be 50 when set explicitly")
        #expect(abs(config.convergenceThreshold - 1e-8) < 1e-15, "convergenceThreshold should remain default")
        #expect(config.isSubspaceExpansionEnabled == false, "isSubspaceExpansionEnabled should remain default")
        #expect(abs(config.noiseStrength - 0.0) < 1e-15, "noiseStrength should remain default")
    }

    @Test("Custom convergenceThreshold configuration")
    func customConvergenceThreshold() {
        let config = DMRGConfiguration(convergenceThreshold: 1e-12)

        #expect(config.maxSweeps == 20, "maxSweeps should remain default")
        #expect(abs(config.convergenceThreshold - 1e-12) < 1e-18, "convergenceThreshold should be 1e-12 when set explicitly")
        #expect(config.isSubspaceExpansionEnabled == false, "isSubspaceExpansionEnabled should remain default")
        #expect(abs(config.noiseStrength - 0.0) < 1e-15, "noiseStrength should remain default")
    }

    @Test("Configuration with subspace expansion enabled")
    func isSubspaceExpansionEnabledEnabled() {
        let config = DMRGConfiguration(isSubspaceExpansionEnabled: true, noiseStrength: 1e-4)

        #expect(config.maxSweeps == 20, "maxSweeps should remain default")
        #expect(abs(config.convergenceThreshold - 1e-8) < 1e-15, "convergenceThreshold should remain default")
        #expect(config.isSubspaceExpansionEnabled == true, "isSubspaceExpansionEnabled should be true when enabled")
        #expect(abs(config.noiseStrength - 1e-4) < 1e-10, "noiseStrength should be 1e-4 when set explicitly")
    }

    @Test("Fully custom configuration")
    func fullyCustomConfiguration() {
        let config = DMRGConfiguration(
            maxSweeps: 100,
            convergenceThreshold: 1e-10,
            isSubspaceExpansionEnabled: true,
            noiseStrength: 1e-5,
        )

        #expect(config.maxSweeps == 100, "maxSweeps should be 100")
        #expect(abs(config.convergenceThreshold - 1e-10) < 1e-16, "convergenceThreshold should be 1e-10")
        #expect(config.isSubspaceExpansionEnabled == true, "isSubspaceExpansionEnabled should be true")
        #expect(abs(config.noiseStrength - 1e-5) < 1e-11, "noiseStrength should be 1e-5")
    }

    @Test("Configurations with same values are equal")
    func configurationsWithSameValuesEqual() {
        let config1 = DMRGConfiguration(maxSweeps: 30, convergenceThreshold: 1e-9)
        let config2 = DMRGConfiguration(maxSweeps: 30, convergenceThreshold: 1e-9)

        #expect(config1 == config2, "Configurations with identical values should be equal")
    }

    @Test("Configurations with different maxSweeps are not equal")
    func configurationsWithDifferentMaxSweepsNotEqual() {
        let config1 = DMRGConfiguration(maxSweeps: 20)
        let config2 = DMRGConfiguration(maxSweeps: 30)

        #expect(config1 != config2, "Configurations with different maxSweeps should not be equal")
    }

    @Test("Configurations with different convergenceThreshold are not equal")
    func configurationsWithDifferentConvergenceThresholdNotEqual() {
        let config1 = DMRGConfiguration(convergenceThreshold: 1e-8)
        let config2 = DMRGConfiguration(convergenceThreshold: 1e-10)

        #expect(config1 != config2, "Configurations with different convergenceThreshold should not be equal")
    }

    @Test("Configurations with different isSubspaceExpansionEnabled are not equal")
    func configurationsWithDifferentSubspaceExpansionNotEqual() {
        let config1 = DMRGConfiguration(isSubspaceExpansionEnabled: false)
        let config2 = DMRGConfiguration(isSubspaceExpansionEnabled: true)

        #expect(config1 != config2, "Configurations with different isSubspaceExpansionEnabled should not be equal")
    }

    @Test("Configurations with different noiseStrength are not equal")
    func configurationsWithDifferentNoiseStrengthNotEqual() {
        let config1 = DMRGConfiguration(noiseStrength: 0.0)
        let config2 = DMRGConfiguration(noiseStrength: 1e-4)

        #expect(config1 != config2, "Configurations with different noiseStrength should not be equal")
    }

    @Test("Default configurations are equal")
    func defaultConfigurationsEqual() {
        let config1 = DMRGConfiguration()
        let config2 = DMRGConfiguration()

        #expect(config1 == config2, "Two default configurations should be equal")
    }

    @Test("Configuration with minimal maxSweeps")
    func minimalMaxSweeps() {
        let config = DMRGConfiguration(maxSweeps: 1)

        #expect(config.maxSweeps == 1, "maxSweeps should accept minimum value of 1")
    }

    @Test("Configuration with zero noiseStrength")
    func zeroNoiseStrength() {
        let config = DMRGConfiguration(isSubspaceExpansionEnabled: true, noiseStrength: 0.0)

        #expect(config.isSubspaceExpansionEnabled == true, "isSubspaceExpansionEnabled should be true")
        #expect(abs(config.noiseStrength - 0.0) < 1e-15, "noiseStrength should be exactly 0.0")
    }

    @Test("Configuration with very small convergenceThreshold")
    func verySmallConvergenceThreshold() {
        let config = DMRGConfiguration(convergenceThreshold: 1e-15)

        #expect(abs(config.convergenceThreshold - 1e-15) < 1e-20, "convergenceThreshold should accept very small values like 1e-15")
    }

    @Test("Configuration with large maxSweeps")
    func largeMaxSweeps() {
        let config = DMRGConfiguration(maxSweeps: 1000)

        #expect(config.maxSweeps == 1000, "maxSweeps should accept large values like 1000")
    }
}

/// Test suite for DMRGSweepDirection enum.
/// Validates enum case existence, Equatable conformance,
/// and distinct identity of leftToRight and rightToLeft directions.
@Suite("DMRGSweepDirection")
struct DMRGSweepDirectionTests {
    @Test("LeftToRight case exists")
    func leftToRightCaseExists() {
        let direction = DMRGSweepDirection.leftToRight

        #expect(direction == .leftToRight, "leftToRight case should be accessible")
    }

    @Test("RightToLeft case exists")
    func rightToLeftCaseExists() {
        let direction = DMRGSweepDirection.rightToLeft

        #expect(direction == .rightToLeft, "rightToLeft case should be accessible")
    }

    @Test("LeftToRight and rightToLeft are not equal")
    func directionsAreNotEqual() {
        let left = DMRGSweepDirection.leftToRight
        let right = DMRGSweepDirection.rightToLeft

        #expect(left != right, "leftToRight and rightToLeft should be distinct directions")
    }

    @Test("Same directions are equal")
    func sameDirectionsEqual() {
        let left1 = DMRGSweepDirection.leftToRight
        let left2 = DMRGSweepDirection.leftToRight
        let right1 = DMRGSweepDirection.rightToLeft
        let right2 = DMRGSweepDirection.rightToLeft

        #expect(left1 == left2, "Two leftToRight directions should be equal")
        #expect(right1 == right2, "Two rightToLeft directions should be equal")
    }

    @Test("Direction can be toggled in sweep pattern")
    func directionTogglePattern() {
        var direction = DMRGSweepDirection.leftToRight

        direction = (direction == .leftToRight) ? .rightToLeft : .leftToRight
        #expect(direction == .rightToLeft, "Direction should toggle from leftToRight to rightToLeft")

        direction = (direction == .leftToRight) ? .rightToLeft : .leftToRight
        #expect(direction == .leftToRight, "Direction should toggle from rightToLeft back to leftToRight")
    }

    @Test("Directions can be stored in array")
    func directionsInArray() {
        let directions: [DMRGSweepDirection] = [.leftToRight, .rightToLeft, .leftToRight]

        #expect(directions.count == 3, "Array should contain 3 directions")
        #expect(directions[0] == .leftToRight, "First direction should be leftToRight")
        #expect(directions[1] == .rightToLeft, "Second direction should be rightToLeft")
        #expect(directions[2] == .leftToRight, "Third direction should be leftToRight")
    }
}
