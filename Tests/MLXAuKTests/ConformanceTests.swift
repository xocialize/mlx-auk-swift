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
            XCTAssertEqual(edit?.operations, [.emotion, .style, .denoise, .dereverberate, .replaceWords, .insertWords,
                                              .removeWords, .speed, .pitch, .volume, .timbre, .deAccent, .addNonverbal,
                                              .removeNonverbal])
            XCTAssertEqual(edit?.emotionLabels.count, 8)
            XCTAssertEqual(edit?.styleLabels, ["whisper", "normal"])
            XCTAssertEqual(edit?.nonverbalSounds, ["breath", "laugh", "cough", "humming"])
            XCTAssertEqual(edit?.speedFactors, [0.5, 0.75, 1.25, 1.5, 2.0])
            XCTAssertEqual(edit?.pitchSemitones, [-3, -2, -1, 1, 2, 3])
            XCTAssertEqual(edit?.volumeDecibels, [-15, -10, -5, 5, 10, 15])
            XCTAssertEqual(edit?.transcript, .unused)
            XCTAssertEqual(edit?.supportsTargetDuration, true)
            XCTAssertEqual(edit?.maxInputSeconds, 30)
            let editSurface = m.surfaces.first { $0.capability == .speechEdit }
            XCTAssertEqual(editSurface?.parameters.first { $0.name == "transcript" }?.required, false)
            XCTAssertNotNil(editSurface?.parameters.first { $0.name == "targetDuration" })
            let tts = m.surfaces.first { $0.capability == .tts }?.ttsControls
            XCTAssertEqual(tts?.emotionModes, [.textDescription])
            XCTAssertEqual(tts?.supportsTargetDuration, true)
        }
    }

    /// C-memory — three tiers, at least what `auk-gates validate` measured through the engine: the resident floor
    /// (the larger of the 2026-10-18 and 2026-11-01 runs), and resident + activation at least the in-request peak the
    /// 5 ms sampler and kernel ledger saw (AB-T-0414).
    func testFootprintsAreTheMeasuredTiers() {
        for (m, resident, peak) in [(AuKPackage.manifest, [12_750_000_000, 9_460_000_000, 7_970_000_000],
                                     [14_920_000_000, 11_750_000_000, 10_040_000_000]),
                                    (AuKFlashPackage.manifest, [12_770_000_000, 9_490_000_000, 7_920_000_000],
                                     [14_890_000_000, 11_610_000_000, 9_830_000_000])] {
            let f = m.requirements.footprints
            XCTAssertEqual(f.map(\.quant), [.bf16, .int8, .int4])
            for (fp, (r, p)) in zip(f, zip(resident, peak)) {
                XCTAssertGreaterThanOrEqual(fp.residentBytes, UInt64(r))
                XCTAssertGreaterThanOrEqual(fp.residentBytes + fp.peakActivationBytes, UInt64(p))
                XCTAssertGreaterThanOrEqual(fp.peakActivationBytes, 2_560_000_000)
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
        XCTAssertEqual(AuKSurfaces.language(for: .style("whisper")), .zh)
        XCTAssertEqual(AuKSurfaces.language(for: .style("normal")), .en)
        XCTAssertEqual(try AuKSurfaces.instruction(for: .denoise), .denoise)
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .emotion("remove")))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .style("shout")))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .trimSilence))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .paralinguistic(targetTranscript: "a [Laughter]")))
        // 1.59.0 kinds
        XCTAssertEqual(try AuKSurfaces.instruction(for: .dereverberate), .dereverberate)
        XCTAssertEqual(try AuKSurfaces.instruction(for: .replaceWords(original: "a", new: "b")), .replace(original: "a", new: "b"))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .insertWords(text: "very", anchor: "good", before: true)),
                       .insert(text: "very", anchor: "good", before: true))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .removeWords(text: "x", anchor: nil, before: false)),
                       .remove(text: "x", anchor: nil, before: false))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .speed(factor: 1.5)), .speed(factor: 1.5))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .pitch(semitones: -2)), .pitch(semitones: -2))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .volume(dB: 10)), .volume(dB: 10))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .timbre(description: "a deep voice")), .timbre(description: "a deep voice"))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .deAccent), .deAccent)
        XCTAssertEqual(try AuKSurfaces.instruction(for: .addNonverbal(sound: "cough", atEnd: true)), .addNonverbal(sound: "cough", atEnd: true))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .removeNonverbal(sound: "breath")), .removeNonverbal("breaths"))
        XCTAssertEqual(try AuKSurfaces.instruction(for: .removeNonverbal(sound: "humming")), .removeNonverbal("humming"))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .removeNonverbal(sound: nil)))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .addNonverbal(sound: "sneeze", atEnd: false)))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .speed(factor: 1.1)))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .pitch(semitones: 4)))
        XCTAssertThrowsError(try AuKSurfaces.instruction(for: .volume(dB: 3)))
    }

    /// The declaration's own refusals (`SpeechEditControls.refusal(for:)`), which the package runs first.
    func testDeclaredRefusals() {
        let controls = AuKSurfaces.speechEditControls()
        func refused(_ edit: SpeechEditOperation, transcript: String = "", seconds: Double? = nil) -> Bool {
            controls.refusal(for: SpeechEditRequest(audio: clip, transcript: transcript, edit: edit, targetDuration: seconds)) != nil
        }
        XCTAssertFalse(refused(.emotion("happy")))                                     // no transcript needed
        XCTAssertFalse(refused(.speed(factor: 2.0), seconds: 4))
        XCTAssertTrue(refused(.speed(factor: 1.1)))
        XCTAssertTrue(refused(.pitch(semitones: 0)))
        XCTAssertTrue(refused(.volume(dB: 20)))
        XCTAssertTrue(refused(.addNonverbal(sound: "sneeze", atEnd: true)))
        XCTAssertTrue(refused(.trimSilence))
        XCTAssertTrue(refused(.denoise, seconds: -1))
    }

    /// An unloaded package answers `notLoaded` before it judges the request, as for 1.50.0 edits.
    func testUnloadedPackageAnswersNotLoadedBeforeRefusing() async {
        let package = AuKPackage(configuration: AuKConfiguration())
        do {
            _ = try await package.run(SpeechEditRequest(audio: clip, edit: .speed(factor: 1.1)))
            XCTFail("expected a refusal")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded first, got \(error)") }
        } catch {
            XCTFail("expected PackageError, got \(error)")
        }
    }

    func testEditLengths() {
        func seconds(_ edit: SpeechEditOperation, transcript: String = "", target: Double? = nil) -> Double? {
            AuKSurfaces.editSeconds(for: SpeechEditRequest(audio: clip, transcript: transcript, edit: edit, targetDuration: target),
                                    sourceSeconds: 6)
        }
        XCTAssertNil(seconds(.emotion("happy")))                                        // the source's own length
        XCTAssertNil(seconds(.addNonverbal(sound: "laugh", atEnd: true)))
        XCTAssertEqual(seconds(.emotion("happy"), target: 5), 5)                         // a targetDuration wins
        XCTAssertEqual(seconds(.speed(factor: 1.5)), 4)
        XCTAssertEqual(seconds(.speed(factor: 0.5)), 12)
        // with a transcript: upstream's byte ratio (6 s × 7 bytes "a horse" / 5 bytes "a cat")
        XCTAssertEqual(seconds(.replaceWords(original: "cat", new: "horse"), transcript: "a cat") ?? 0,
                       6 * 7 / 5, accuracy: 1e-9)
        // without one: the speaking-rate estimate of the changed words
        XCTAssertEqual(seconds(.insertWords(text: "fourteen chars", anchor: "x", before: true)) ?? 0, 7, accuracy: 1e-9)
        XCTAssertEqual(seconds(.removeWords(text: "fourteen chars", anchor: nil, before: false)) ?? 0, 5, accuracy: 1e-9)
        XCTAssertEqual(seconds(.removeWords(text: String(repeating: "x", count: 200), anchor: nil, before: false)), 0.5)
    }

    func testEditedTranscripts() {
        func edited(_ edit: SpeechEditOperation, _ transcript: String) -> String {
            AuKSurfaces.editedTranscript(for: edit, transcript: transcript)
        }
        XCTAssertEqual(edited(.replaceWords(original: "cat", new: "dog"), "the cat sat"), "the dog sat")
        XCTAssertEqual(edited(.insertWords(text: "black", anchor: "cat", before: true), "the cat sat"), "the black cat sat")
        XCTAssertEqual(edited(.insertWords(text: "down", anchor: "sat", before: false), "the cat sat."), "the cat sat down.")
        XCTAssertEqual(edited(.insertWords(text: "黑", anchor: "猫", before: true), "那只猫"), "那只黑猫")
        XCTAssertEqual(edited(.removeWords(text: "very", anchor: nil, before: false), "a very good day"), "a good day")
        XCTAssertEqual(edited(.removeWords(text: "very", anchor: "bad", before: true), "very good, very bad"), "very good, bad")
        XCTAssertEqual(edited(.removeWords(text: "very", anchor: "good", before: false), "very good, very bad"), "very good, bad")
        XCTAssertEqual(edited(.replaceWords(original: "absent", new: "x"), "the cat sat"), "")    // words not in the transcript
        XCTAssertEqual(edited(.replaceWords(original: "cat", new: "dog"), ""), "")                 // no transcript given
        XCTAssertEqual(edited(.emotion("happy"), " the cat sat "), "the cat sat")                  // echoed
    }

    func testEditLanguage() {
        XCTAssertEqual(AuKSurfaces.language(for: .replaceWords(original: "论文", new: "报告")), .zh)
        XCTAssertEqual(AuKSurfaces.language(for: .replaceWords(original: "cat", new: "dog")), .en)
        XCTAssertEqual(AuKSurfaces.language(for: .timbre(description: "低沉的男声")), .zh)
        XCTAssertEqual(AuKSurfaces.language(for: .emotion("happy")), .en)
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
        XCTAssertEqual(AukInstruction.instructTTS(voiceDescription: "A warm voice.", text: "Hi").text(),
                       "Based on the following description: \"A warm voice.\", generate speech content \"Hi\".")
        XCTAssertEqual(AukInstruction.speed(factor: 1.5).text(), "Adjust the speech speed to 1.5x.")
        XCTAssertEqual(AukInstruction.speed(factor: 2.0).text(), "Adjust the speech speed to 2.0x.")
        XCTAssertEqual(AukInstruction.replace(original: "a", new: "b").text(), "Replace 'a' with 'b'.")
        XCTAssertEqual(AukInstruction.emotion("happy").text(), "Change the emotion to happy.")
        XCTAssertEqual(AukInstruction.pitch(semitones: -2).text(), "Lower the pitch by 2 semitones.")
        XCTAssertEqual(AukInstruction.volume(dB: 10).text(), "Increase the volume by 10 dB.")
        XCTAssertEqual(AukInstruction.enhance.text(),
                       "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.")
        XCTAssertEqual(AukInstruction.removeNonverbal("breaths").text(), "Remove all breaths from the audio.")
        XCTAssertEqual(AukInstruction.addNonverbal(sound: "laugh", atEnd: true).text(), "Add a laugh at the end of the speech.")
        XCTAssertEqual(AukInstruction.addNonverbal(sound: "humming", atEnd: false).text(),
                       "Add humming at the beginning of the speech.")
        XCTAssertEqual(AukInstruction.replace(original: "论文", new: "报告").text(.zh), "把‘论文’改成‘报告’")
        XCTAssertEqual(AukInstruction.remove(text: "很", anchor: nil, before: false).text(.zh), "删掉‘很’")
        XCTAssertEqual(AukInstruction.remove(text: "很", anchor: "好", before: true).text(.zh), "删掉‘好’前面的‘很’")
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
