// ConformanceTests.swift — the offline conformance gate for MLXAuK: manifest (C1 / C7 / C8 / provenance), the restricted
// licence posture (AB-D-0153), the MAT gate per tier, the CAN gate, the unloaded refusal, and the request plane
// (TTS planning, speechEdit operations → AuK instructions). Nothing here evals a kernel: the live run is
// `auk-gates validate` (GPU).

import Foundation
import MLX
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import AuKCore
@testable import MLXAuK

final class ManifestConformanceTests: XCTestCase {

    /// C7/C8 — the required encoder makes the declared weight licence LicenseRef-Qwen-Research (the DiT + VAE MIT ride
    /// along); never allowlisted, so the permissive policies refuse it by construction.
    func testLicenseIsQwenResearchAndRefusedByPermissivePolicies() {
        for m in [AuKPackage.manifest, AuKFlashPackage.manifest] {
            XCTAssertEqual(m.license.weightLicense, .qwenResearch)
            XCTAssertEqual(m.license.weightLicense, SPDXLicense("LicenseRef-Qwen-Research"))
            XCTAssertEqual(m.license.additionalWeightLicenses, [.mit])
            XCTAssertEqual(m.license.portCodeLicense, .mit)
            XCTAssertFalse(SPDXLicense.permissiveAllowlist.contains(m.license.weightLicense))
            XCTAssertFalse(LicensePolicy.permissiveOnly.evaluate(m.license).isAdmitted)
        }
    }

    /// C1 — both packages serve tts + speechEdit, with declared controls.
    func testCapabilitiesAndDeclarations() {
        for m in [AuKPackage.manifest, AuKFlashPackage.manifest] {
            XCTAssertEqual(Set(m.capabilities), [.tts, .speechEdit])
            let edit = m.surfaces.first { $0.capability == .speechEdit }?.speechEditControls
            XCTAssertEqual(edit?.operations, [.emotion, .style, .denoise])
            XCTAssertEqual(edit?.emotionLabels.count, 8)
            XCTAssertEqual(edit?.styleLabels, ["whisper", "normal"])
            XCTAssertEqual(edit?.maxInputSeconds, 30)
            let tts = m.surfaces.first { $0.capability == .tts }?.ttsControls
            XCTAssertEqual(tts?.emotionModes, [.textDescription])
            XCTAssertEqual(tts?.supportsTargetDuration, true)
        }
    }

    /// C-memory — three tiers, at least what `auk-gates validate` measured through the engine.
    func testFootprintsAreTheMeasuredTiers() {
        for (m, measured) in [(AuKPackage.manifest, [12_750_000_000, 9_460_000_000, 7_630_000_000]),
                              (AuKFlashPackage.manifest, [12_760_000_000, 9_490_000_000, 7_900_000_000])] {
            let f = m.requirements.footprints
            XCTAssertEqual(f.map(\.quant), [.bf16, .int8, .int4])
            for (fp, bytes) in zip(f, measured) {
                XCTAssertGreaterThanOrEqual(fp.residentBytes, UInt64(bytes))
                XCTAssertGreaterThanOrEqual(fp.peakActivationBytes, 1_030_000_000)
            }
        }
    }

    func testProvenancePinsUpstream() {
        XCTAssertEqual(AuKPackage.manifest.provenance.sourceRepo, "tencent/AuK")
        XCTAssertEqual(AuKFlashPackage.manifest.provenance.sourceRepo, "tencent/AuK-Flash")
        XCTAssertEqual(AuKPackage.manifest.provenance.revision.count, 40)
        XCTAssertEqual(AuKFlashPackage.manifest.provenance.revision.count, 40)
    }
}

final class MaterializationConformanceTests: XCTestCase {

    func testMaterializationGatePerTier() throws {
        for variant in [AuKVariant.base, .flash] {
            for quant in [Quant.bf16, .int8, .int4, .fp32] {
                let satisfied = try satisfiedConfiguration(variant: variant, quant: quant)
                defer {
                    try? FileManager.default.removeItem(at: satisfied.modelDirectory!)
                    try? FileManager.default.removeItem(at: satisfied.encoderDirectory!)
                }
                let report = MaterializationConformance.check(
                    freshConfiguration: AuKConfiguration(variant: variant, quant: quant, modelsRootDirectory: emptyStoreRoot()),
                    satisfiedConfiguration: satisfied)
                XCTAssertTrue(report.passed, "\(variant) \(quant): \(report.summary)")
            }
        }
    }

    /// Two sources per tier, named by fleet convention: the MIT model bundle and the Qwen-Research encoder bundle.
    func testPublishedRepos() {
        XCTAssertEqual(AuKConfiguration(variant: .base).weightSources.map(\.repo),
                       ["mlx-community/AuK-bf16", "mlx-community/Qwen2.5-Omni-3B-Thinker-bf16"])
        XCTAssertEqual(AuKConfiguration(variant: .flash, quant: .int8).weightSources.map(\.repo),
                       ["mlx-community/AuK-Flash-bf16", "mlx-community/Qwen2.5-Omni-3B-Thinker-8bit"])
        XCTAssertEqual(AuKConfiguration(variant: .base, quant: .int4).weightSources.map(\.repo),
                       ["mlx-community/AuK-bf16", "mlx-community/Qwen2.5-Omni-3B-Thinker-4bit"])
        XCTAssertEqual(AuKConfiguration(variant: .flash, quant: .fp32).weightSources.map(\.repo),
                       ["mlx-community/AuK-Flash-bf16", "mlx-community/Qwen2.5-Omni-3B-Thinker-bf16"])
    }

    func testCodableExcludesEnvironmentURLs() throws {
        let c = AuKConfiguration(variant: .base, quant: .int8, modelDirectory: URL(fileURLWithPath: "/tmp/x"),
                                 encoderDirectory: URL(fileURLWithPath: "/tmp/z"), modelsRootDirectory: URL(fileURLWithPath: "/tmp/y"))
        let decoded = try JSONDecoder().decode(AuKConfiguration.self, from: JSONEncoder().encode(c))
        XCTAssertNil(decoded.modelDirectory)
        XCTAssertNil(decoded.encoderDirectory)
        XCTAssertNil(decoded.modelsRootDirectory)
        XCTAssertEqual(decoded.variant, .base)
        XCTAssertEqual(decoded.quant, .int8)
        XCTAssertTrue(decoded.warmUp)
    }

    private func emptyStoreRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("auk-empty-store-\(UUID().uuidString)")
    }

    private func satisfiedConfiguration(variant: AuKVariant, quant: Quant) throws -> AuKConfiguration {
        let fm = FileManager.default
        let m = fm.temporaryDirectory.appendingPathComponent("auk-model-\(UUID().uuidString)")
        let e = fm.temporaryDirectory.appendingPathComponent("auk-encoder-\(UUID().uuidString)")
        try fm.createDirectory(at: m, withIntermediateDirectories: true)
        try fm.createDirectory(at: e, withIntermediateDirectories: true)
        try Data().write(to: m.appending(path: "dit.safetensors"))
        try Data().write(to: e.appending(path: "encoder.safetensors"))
        return AuKConfiguration(variant: variant, quant: quant, modelDirectory: m, encoderDirectory: e)
    }
}

final class CancellationConformanceTests: XCTestCase {

    private func probe() -> SpeechEditRequest {
        SpeechEditRequest(audio: Audio(format: .wav, data: Data([1, 2, 3]), sampleRate: 24_000, channels: 1),
                          transcript: "cancellation probe", edit: .emotion("happy"))
    }

    /// CAN-1/CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` unchanged, even unloaded.
    func testPreCancelledRunPropagatesCancellation() async {
        for package in [AuKPackage(configuration: AuKConfiguration()) as any ModelPackage,
                        AuKFlashPackage(configuration: AuKConfiguration())] {
            let report = await CancellationConformance.checkRun(package: package, request: probe())
            XCTAssertTrue(report.passed, report.summary)
        }
    }

    /// CAN-3 — a checkpoint per flow step (RunProgress `.denoise` on the same seam) and at every stage boundary.
    func testCheckpointCadence() {
        for m in [AuKPackage.manifest, AuKFlashPackage.manifest] {
            let report = CancellationConformance.checkCadence(
                manifest: m, posture: .cadence([.init(phase: .denoise, unit: .step, reportsRunProgress: true)]))
            XCTAssertTrue(report.passed, report.summary)
        }
    }
}

final class RequestPlaneTests: XCTestCase {

    private let clip = Audio(format: .wav, data: Data([1, 2, 3]), sampleRate: 24_000, channels: 1)

    func testUnloadedPackageRejectsLegibly() async {
        let package = AuKFlashPackage(configuration: AuKConfiguration())
        do {
            _ = try await package.run(SpeechEditRequest(audio: clip, transcript: "x", edit: .denoise))
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded, got \(error)") }
        } catch {
            XCTFail("expected PackageError, got \(error)")
        }
    }

    func testSpeechEditOperationsMapOntoInstructions() throws {
        XCTAssertEqual(try AuKSurfaces.instruction(for: .emotion("happy")), .emotion("happy"))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .style("whisper")), .toWhisper)
        XCTAssertEqual(try AuKSurfaces.instruction(for: .style("normal")), .fromWhisper)
        XCTAssertEqual(try AuKSurfaces.instruction(for: .denoise), .denoise)
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .emotion("remove")))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .style("shout")))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .trimSilence))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .paralinguistic(targetTranscript: "a [Laughter]")))
    }

    func testTTSPlanning() throws {
        let clone = try AuKSurfaces.plan(TTSRequest(text: "Hello there.", voice: VoiceSelector(.referenceAudio(clip)),
                                                    targetDuration: 2))
        XCTAssertEqual(clone.0, .zeroShotTTS(text: "Hello there."))
        XCTAssertEqual(clone.seconds, 2)
        let design = try AuKSurfaces.plan(TTSRequest(text: "Welcome home.", emotion: .textDescription("A warm voice.")))
        XCTAssertEqual(design.0, .instructTTS(voiceDescription: "A warm voice.", text: "Welcome home."))
        XCTAssertNotNil(design.seconds)
        XCTAssertThrowsError(try AuKSurfaces.plan(TTSRequest(text: "x")))                                  // no voice at all
        XCTAssertThrowsError(try AuKSurfaces.plan(TTSRequest(text: "x", voice: VoiceSelector(.named("alloy")))))
        XCTAssertThrowsError(try AuKSurfaces.plan(TTSRequest(text: "x", emotion: .categorical("happy"))))
        XCTAssertThrowsError(try AuKSurfaces.plan(TTSRequest(text: "x", voice: VoiceSelector(.referenceAudio(clip)),
                                                             emotion: .textDescription("a voice"))))
    }

    /// The canonical cookbook phrasings, verbatim.
    func testInstructionTemplates() {
        XCTAssertEqual(AukInstruction.zeroShotTTS(text: "Hi").text(), "Say the following with the same voice: \"Hi\"")
        XCTAssertEqual(AukInstruction.speed(factor: 1.5).text(), "Adjust the speech speed to 1.5x.")
        XCTAssertEqual(AukInstruction.speed(factor: 2.0).text(), "Adjust the speech speed to 2.0x.")
        XCTAssertEqual(AukInstruction.replace(original: "a", new: "b").text(), "Replace 'a' with 'b'.")
        XCTAssertEqual(AukInstruction.emotion("happy").text(), "Change the emotion to happy.")
        XCTAssertEqual(AukInstruction.pitch(semitones: -2).text(), "Lower the pitch by 2 semitones.")
        XCTAssertEqual(AukInstruction.volume(dB: 10).text(), "Increase the volume by 10 dB.")
        XCTAssertEqual(AukInstruction.enhance.text(),
                       "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.")
    }
}

final class CoreStructuralTests: XCTestCase {
    func testResampleIdentity() { XCTAssertEqual(Resample.sinc([1, 2, 3], from: 24000, to: 24000), [1, 2, 3]) }

    func testAudioTokenCount() {
        XCTAssertEqual(Qwen2_5OmniAudioEncoder.outputLength(melFrames: 633), 158)   // the emotion golden
    }

    func testFlashGridAndBaseSway() {
        let flash = SamplingConfig(nfe: 4, cfg_strength: 0, sway_sampling_coef: nil,
                                   t_grid: [0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1])
        XCTAssertEqual(flash.timeGrid().count, 5)
        let base = SamplingConfig(nfe: 32, cfg_strength: 2, sway_sampling_coef: -1, t_grid: nil).timeGrid()
        XCTAssertEqual(base.count, 33)
        XCTAssertEqual(base.first!, 0, accuracy: 1e-7)
        XCTAssertEqual(base.last!, 1, accuracy: 1e-6)
        XCTAssertLessThan(base[1], 1.0 / 32)   // sway −1 front-loads the small steps
    }
}
