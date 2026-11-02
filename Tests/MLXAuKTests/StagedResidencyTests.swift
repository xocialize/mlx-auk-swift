// StagedResidencyTests.swift — AB-T-0414: the staged mode's lifetime and configuration contract, without weights.
// Bit identity against resident mode needs the real bundles and the GPU: `auk-gates staged` (MEASUREMENTS.md).

import Foundation
import MLX
import XCTest

@testable import AuKCore
@testable import MLXAuK

final class StagedResidencyTests: XCTestCase {
    final class Probe { let id = UUID() }

    /// Resident: one value for the stage's lifetime, the same instance on every use, never rebuilt.
    func testResidentStageHoldsOneValue() throws {
        let probe = Probe()
        let stage = AukStage(resident: probe)
        XCTAssertTrue(stage.held === probe)
        let a = try stage.use { $0.id }, b = try stage.use { $0.id }
        XCTAssertEqual(a, b)
        XCTAssertEqual(stage.lastBuildSeconds, 0)
    }

    /// Staged: a fresh value per use, nothing held between uses, and the value is gone once `use` returns — the
    /// property the peak depends on.
    func testStagedStageBuildsPerUseAndReleases() throws {
        var builds = 0
        weak var last: Probe?
        let stage = AukStage<Probe>(staged: { builds += 1; return Probe() })
        XCTAssertNil(stage.held)
        let a = try stage.use { p -> UUID in last = p; return p.id }
        XCTAssertNil(last, "the staged value outlived its use")
        let b = try stage.use { $0.id }
        XCTAssertEqual(builds, 2)
        XCTAssertNotEqual(a, b)
        XCTAssertNil(stage.held)
    }

    func testStagedBuildErrorPropagates() {
        let stage = AukStage<Probe>(staged: { throw AuKError.missingFile("encoder.safetensors") })
        XCTAssertThrowsError(try stage.use { _ in () })
    }

    /// Default resident; `residency` round-trips; a configuration encoded before the field existed decodes resident.
    func testResidencyConfiguration() throws {
        XCTAssertEqual(AuKConfiguration().residency, .resident)
        let staged = AuKConfiguration(variant: .base, quant: .int8, residency: .staged)
        let decoded = try JSONDecoder().decode(AuKConfiguration.self, from: JSONEncoder().encode(staged))
        XCTAssertEqual(decoded.residency, .staged)
        XCTAssertEqual(decoded.quant, .int8)
        let legacy = #"{"variant":"flash","quant":"bf16","warmUp":true}"#.data(using: .utf8)!
        XCTAssertEqual(try JSONDecoder().decode(AuKConfiguration.self, from: legacy).residency, .resident)
    }

    /// The staged pipeline's fusion head reads only `layer_weights` / `layer_scale` from dit.safetensors, at their
    /// stored dtype — the same tensors the resident CFMEdit holds. CPU stream: no GPU outside the AuK GPU lock.
    func testFusionLoadsOnlyTheFusionTensors() throws {
        try Device.withDefaultDevice(.cpu) { try fusionLoad() }
    }

    private func fusionLoad() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("auk-fusion-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let lw = MLXArray((0 ..< 36).map { Float($0) / 10 }), ls = MLXArray([Float(0.5)])
        try save(arrays: ["layer_weights": lw, "layer_scale": ls,
                          "transformer.x_embedder.weight": MLXArray.zeros([4, 4]).asType(.bfloat16)], url: url)
        let fusion = try AukFusion.load(url)
        XCTAssertEqual(fusion.layerWeights.dtype, .float32)
        XCTAssertEqual(fusion.layerWeights.asArray(Float.self), lw.asArray(Float.self))
        XCTAssertEqual(fusion.layerScale.asArray(Float.self), [0.5])

        let partial = FileManager.default.temporaryDirectory.appendingPathComponent("auk-fusion-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: partial) }
        try save(arrays: ["layer_weights": lw], url: partial)
        XCTAssertThrowsError(try AukFusion.load(partial))
    }
}
