// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import PackageDescription

let package = Package(
    name: "Aether",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "Aether",
            targets: ["Aether"]
        ),
        .executable(
            name: "AetherDemo",
            targets: ["AetherDemo"]
        ),
    ],
    targets: [
        .target(
            name: "Aether",
            path: "Sources/Aether",
            resources: [
                .process("QuantumEngine/QuantumGPU.metal"),
            ],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK", to: "1"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .define("ACCELERATE_NEW_LAPACK"),
            ]
        ),

        .executableTarget(
            name: "AetherDemo",
            dependencies: ["Aether"],
            path: "Sources/AetherDemo",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .define("ACCELERATE_NEW_LAPACK"),
            ]
        ),

        .testTarget(
            name: "AetherTests",
            dependencies: ["Aether"],
            path: "Tests/AetherTests",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .define("ACCELERATE_NEW_LAPACK"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
