// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation
import Metal

/// Shared Metal utilities for GPU-accelerated quantum computing
///
/// Centralizes common Metal device initialization, library loading, and command buffer
/// creation patterns used across MetalGateApplication and SparseHamiltonian backends.
/// Implements 3-tier fallback strategy for maximum compatibility across deployment
/// environments (compiled metallib, default library, source compilation).
///
/// **Library Loading Strategy**:
/// 1. **Compiled metallib**: Fastest, pre-compiled binary shaders
/// 2. **Default library**: Built-in to app bundle during Xcode build
/// 3. **Source compilation**: Fallback for non-standard deployments (slower first load)
///
/// **Thread Safety**: All methods are stateless and thread-safe. Device/queue creation
/// can be called concurrently from multiple threads without synchronization.
@frozen
public enum MetalUtilities {
    /// Load Metal shader library with 3-tier fallback strategy
    ///
    /// Attempts to load QuantumGPU shader library in order of performance preference:
    /// 1. Pre-compiled `default.metallib` file (fastest, production deployments)
    /// 2. Default library from app bundle (Xcode builds with Metal files)
    /// 3. Runtime compilation from `QuantumGPU.metal` source (development/non-standard)
    ///
    /// **Error Handling**: Returns nil if all three loading strategies fail. Caller must
    /// handle Metal unavailability with CPU fallback.
    ///
    /// - Parameter device: Metal device to create library for
    /// - Returns: Shader library if any loading strategy succeeds, nil otherwise
    @_effects(readonly)
    public static func loadLibrary(device: MTLDevice) -> MTLLibrary? {
        // Strategy 1: Pre-compiled metallib (xcode)
        if let metallibURL = Bundle.module.url(forResource: "default", withExtension: "metallib"),
           let library = try? device.makeLibrary(URL: metallibURL)
        { return library }

        // Strategy 2: Default library
        if let defaultLibrary = device.makeDefaultLibrary() { return defaultLibrary }

        // Strategy 3: Runtime source compilation (CLI)
        guard let resourceURL = Bundle.module.url(forResource: "QuantumGPU", withExtension: "metal"),
              let source = try? String(contentsOf: resourceURL, encoding: .utf8),
              let library = try? device.makeLibrary(source: source, options: nil)
        else { return nil }

        return library
    }

    /// Create Metal command buffer and compute encoder with fallback handling
    ///
    /// Convenience wrapper for the common pattern of creating command buffer + encoder
    /// with guard-let error handling. Reduces boilerplate in GPU kernel dispatch code.
    ///
    /// **Usage Pattern**:
    /// ```swift
    /// guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue)
    /// else {
    ///     // Fallback to CPU implementation
    ///     return cpuFallback()
    /// }
    /// // Configure encoder...
    /// encoder.endEncoding()
    /// commandBuffer.commit()
    /// ```
    ///
    /// - Parameter queue: Metal command queue to create buffer from
    /// - Returns: (command buffer, compute encoder) tuple if creation succeeds, nil otherwise
    @_effects(readonly)
    @inlinable
    public static func createCommandEncoder(
        queue: MTLCommandQueue
    ) -> (commandBuffer: MTLCommandBuffer, encoder: MTLComputeCommandEncoder)? {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }

        return (commandBuffer, encoder)
    }
}
