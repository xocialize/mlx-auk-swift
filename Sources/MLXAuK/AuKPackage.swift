import Foundation
import MLX
import MLXNN
import MLXToolKit
import AuKCore

extension SPDXLicense {
    /// The Qwen RESEARCH LICENSE AGREEMENT (non-commercial; redistribution with the agreement + NOTICE, "Built with
    /// Qwen"). Not an SPDX id, so it takes the `LicenseRef-` convention — the same string qwen-image21-swift declares.
    /// Intentionally absent from `permissiveAllowlist` (AB-D-0153).
    public static let qwenResearch: SPDXLicense = "LicenseRef-Qwen-Research"
}

/// AuK (Tencent Hunyuan) on two canonical surfaces — `tts` and `speechEdit` — as a RESTRICTED engine model
/// (AB-D-0153): the DiT and VAE are MIT, but the required Qwen2.5-Omni-3B encoder is Qwen-Research, so the package
/// declares `LicenseRef-Qwen-Research` and is refused by `.permissiveOnly` / `.permissiveOrAcknowledged` by construction.
///
/// `tts`: zero-shot cloning from `voice: .referenceAudio` (the target length from `targetDuration`, else upstream's
/// byte-ratio estimate from `referenceTranscript`), or instruct TTS from `emotion: .textDescription` with no reference
/// (a voice designed from the description; `targetDuration` or a speaking-rate estimate).
///
/// `speechEdit` (contract 1.59.0, AB-A-0188 / AB-D-0200): `.emotion` (AuK's 8 labels), `.style("whisper")` /
/// `.style("normal")` (whisper conversion both ways), `.denoise`, `.dereverberate`, the content edits
/// (`.replaceWords` / `.insertWords` / `.removeWords`), `.speed` / `.pitch` / `.volume` on AuK's trained grids,
/// `.timbre`, `.deAccent`, and `.addNonverbal` / `.removeNonverbal` of one named sound. AuK conditions on the audio
/// alone (`transcript: .unused`); a transcript given is echoed, or edited for a content edit, and sharpens the length
/// estimate. The output keeps the source's length unless the edit changes it: then `targetDuration`, else a speed
/// edit scales the source by 1 / factor and a content edit estimates from the text (`AuKSurfaces.editSeconds`).
///
/// `metaData` (package-specific, C5): `seed` (int) for TTS — speechEdit takes the request's `seed`; `nfe` and
/// `cfgStrength` (Base only — Flash pins its distilled 4-step recipe).
@InferenceActor
public final class AuKSession {
    let configuration: AuKConfiguration
    var pipeline: AukPipeline?

    nonisolated init(configuration: AuKConfiguration) { self.configuration = configuration }

    func load() async throws {
        guard pipeline == nil else { return }
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw AuKPackageError.missingWeights(
                "sources not materialized: \(missing.map(\.role).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()
        let r = configuration.resolved(storeRoot: storeRoot)
        guard let modelDir = r.modelDirectory, let encDir = r.encoderDirectory else {
            throw AuKPackageError.missingWeights("unresolved bundle directories (no store root)")
        }
        pipeline = try await AukPipeline.load(modelDir: modelDir, encoderDir: encDir, dtype: configuration.dtype,
                                              residency: configuration.residency)
        if configuration.warmUp {
            do {
                _ = try pipeline?.generate(AukRequest(instruction: AukInstruction.instructTTS(
                    voiceDescription: "A calm adult voice.", text: "Hello.").text(), genSeconds: 0.5))
                MLX.Memory.clearCache()
            } catch {
                // a failed warm-up is not a failed load: the first real request pays the compile
            }
        }
    }

    func unload() {
        pipeline = nil
        MLX.Memory.clearCache()
    }

    func render(_ instruction: AukInstruction, language: AukLanguage = .en, audio: (samples: [Float], rate: Int)?,
                seconds: Double?, seed: UInt64, meta: MetaData) throws -> Data {
        guard let pipeline else { throw PackageError.notLoaded }
        var sampling: SamplingConfig? = nil
        if !pipeline.config.is_flash, meta.intValue("nfe") != nil || meta.doubleValue("cfgStrength") != nil {
            var s = pipeline.config.sampling
            if let n = meta.intValue("nfe") { s.nfe = n }
            if let c = meta.doubleValue("cfgStrength") { s.cfg_strength = c }
            sampling = s
        }
        let wave = try pipeline.generate(
            AukRequest(instruction: instruction.text(language), audio: audio?.samples, sampleRate: audio?.rate ?? 24000,
                       genSeconds: seconds, seed: seed, sampling: sampling),
            onStep: { RunProgress.report(.denoise, step: $0, totalSteps: $1) })
        return AuKAudioIO.encodeWAV16(samples: wave, sampleRate: pipeline.sampleRate)
    }
}

public enum AuKSurfaces {
    public nonisolated static let emotionLabels = AukInstruction.emotions
    public nonisolated static let styleLabels = ["whisper", "normal"]
    /// Training utterances are 1–35 s; upstream's ComfyUI node caps source + target at 30 s.
    public nonisolated static let maxInputSeconds: Double = 30
    /// The nonverbal sounds the cookbook names (§4.4: breaths, laughs, coughs; humming in its remove example), as
    /// singular labels: `.addNonverbal` says "a cough", `.removeNonverbal` "all coughs" (`nonverbalPhrase`).
    public nonisolated static let nonverbalSounds = ["breath", "laugh", "cough", "humming"]
    /// Instruct-TTS length estimate when no targetDuration is given (Latin ≈ 14 chars/s, CJK ≈ 4.5 chars/s).
    nonisolated static func estimatedSpeechSeconds(_ text: String) -> Double {
        max(0.8, speechSeconds(text))
    }

    /// The same speaking-rate estimate without a floor — for the length a content edit adds or takes away.
    nonisolated static func speechSeconds(_ text: String) -> Double {
        var cjk = 0, other = 0
        for s in text.unicodeScalars {
            if isCJK(s) { cjk += 1 } else { other += 1 }
        }
        return Double(cjk) / 4.5 + Double(other) / 14.0
    }

    nonisolated static func isCJK(_ s: Unicode.Scalar) -> Bool {
        (0x4E00 ... 0x9FFF).contains(s.value) || (0x3400 ... 0x4DBF).contains(s.value)
    }

    nonisolated static func speechEditControls() -> SpeechEditControls {
        SpeechEditControls(
            operations: [.emotion, .style, .denoise, .dereverberate, .replaceWords, .insertWords, .removeWords,
                         .speed, .pitch, .volume, .timbre, .deAccent, .addNonverbal, .removeNonverbal],
            emotionLabels: emotionLabels, styleLabels: styleLabels, paralinguisticTags: [],
            nonverbalSounds: nonverbalSounds,
            speedFactors: AukInstruction.speedFactors, pitchSemitones: AukInstruction.pitchSteps,
            volumeDecibels: AukInstruction.volumeSteps,
            transcript: .unused, supportsTargetDuration: true, maxInputSeconds: maxInputSeconds)
    }

    nonisolated static func manifest(variant: AuKVariant) -> PackageManifest {
        let name = variant == .flash ? "auk-flash" : "auk"
        let speed = variant == .flash ? "4 distilled steps, no CFG" : "32 steps × CFG 2"
        let fp = variant == .flash ? AuKFootprints.flash : AuKFootprints.base
        return PackageManifest(
            // C7: DiT + VAE MIT (tencent/AuK[-Flash]); the required encoder Qwen/Qwen2.5-Omni-3B is Qwen-Research — the
            // most restrictive component is the declared weight licence. C8: port code MIT.
            license: LicenseDeclaration(weightLicense: .qwenResearch, additionalWeightLicenses: [.mit], portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: variant == .flash ? "tencent/AuK-Flash" : "tencent/AuK",
                                   revision: variant == .flash ? "575b92f0895f75180bf2cbd35f2e176c5732b8ed"
                                                               : "790742b71a4430120daf2b2099192abae449eb9f",
                                   tier: 3),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: fp.bf16Resident, peakActivationBytes: fp.peakActivation),
                    QuantFootprint(quant: .int8, residentBytes: fp.int8Resident, peakActivationBytes: fp.peakActivation),
                    QuantFootprint(quant: .int4, residentBytes: fp.int4Resident, peakActivationBytes: fp.peakActivation),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                TTSContract.descriptor(
                    name: name,
                    summary: "AuK\(variant == .flash ? "-Flash" : "") (Tencent Hunyuan) speech generation (.wav, 24 kHz mono; en / zh; "
                        + "\(speed)): zero-shot cloning from a reference clip, or a voice designed from a text description "
                        + "(emotion.textDescription, no reference). ≤ 30 s. RESEARCH LICENCE (Qwen2.5-Omni encoder): not for "
                        + "commercial use. metaData: seed.",
                    controls: TTSControls(emotionModes: [.textDescription], supportsTargetDuration: true)),
                SpeechEditContract.descriptor(
                    name: name + "-edit",
                    summary: "AuK\(variant == .flash ? "-Flash" : "") (Tencent Hunyuan) speech editor (.wav, 24 kHz mono; en / zh): "
                        + "an existing take in the same voice with words replaced, inserted or removed; faster or slower, higher "
                        + "or lower, louder or quieter on AuK's trained steps; a new emotion, timbre or accent; to or from a "
                        + "whisper; denoised or dereverberated; or a breath, laugh, cough or humming added at an edge or removed. "
                        + "Needs no transcript. Content and speed edits change the length: pass targetDuration to fit a cue. "
                        + "≤ 30 s. The int4 (4-bit encoder) tier loses words in Mandarin edits: use int8 for zh. "
                        + "RESEARCH LICENCE (Qwen2.5-Omni encoder): not for commercial use.",
                    controls: speechEditControls()),
            ]
        )
    }

    /// The instruction for a speechEdit operation.
    nonisolated static func instruction(for op: SpeechEditOperation) throws -> AukInstruction {
        switch op {
        case .emotion(let label):
            guard emotionLabels.contains(label) else {
                throw PackageError.unsupportedRequestFeature("emotion '\(label)' — declared labels: \(emotionLabels.joined(separator: ", "))")
            }
            return .emotion(label)
        case .style(let label):
            switch label {
            case "whisper": return .toWhisper
            case "normal": return .fromWhisper
            default: throw PackageError.unsupportedRequestFeature("style '\(label)' — declared labels: whisper, normal")
            }
        case .denoise:
            return .denoise
        case .dereverberate:
            return .dereverberate
        case .replaceWords(let original, let new):
            return .replace(original: original, new: new)
        case .insertWords(let text, let anchor, let before):
            return .insert(text: text, anchor: anchor, before: before)
        case .removeWords(let text, let anchor, let before):
            return .remove(text: text, anchor: anchor, before: before)
        case .speed(let factor):
            guard AukInstruction.speedFactors.contains(where: { abs($0 - factor) < 1e-9 }) else {
                throw PackageError.unsupportedRequestFeature("speed \(factor)x — AuK's trained factors: 0.5 / 0.75 / 1.25 / 1.5 / 2.0")
            }
            return .speed(factor: factor)
        case .pitch(let semitones):
            guard AukInstruction.pitchSteps.contains(semitones) else {
                throw PackageError.unsupportedRequestFeature("pitch \(semitones) semitones — AuK's trained steps: ±1…3")
            }
            return .pitch(semitones: semitones)
        case .volume(let dB):
            guard AukInstruction.volumeSteps.contains(dB) else {
                throw PackageError.unsupportedRequestFeature("volume \(dB) dB — AuK's trained steps: ±5 / 10 / 15")
            }
            return .volume(dB: dB)
        case .timbre(let description):
            return .timbre(description: description)
        case .deAccent:
            return .deAccent
        case .addNonverbal(let sound, let atEnd):
            return .addNonverbal(sound: try nonverbalPhrase(sound, plural: false), atEnd: atEnd)
        case .removeNonverbal(let sound?):
            return .removeNonverbal(try nonverbalPhrase(sound, plural: true))
        case .removeNonverbal(nil):
            // the cookbook's remove template names one kind of sound; "every sound" has no trained phrasing
            throw PackageError.unsupportedRequestFeature(
                "removeNonverbal needs a sound — AuK removes one kind at a time: \(nonverbalSounds.joined(separator: ", "))")
        case .paralinguistic, .trimSilence:
            throw PackageError.unsupportedRequestFeature("speech edit kind not declared by AuK")
        @unknown default:
            throw PackageError.unsupportedRequestFeature("speech edit kind not known to this package")
        }
    }

    /// A declared nonverbal label in the cookbook's words: "cough" (add) / "coughs" (remove); a mass noun in -ing
    /// ("humming") stays as it is.
    nonisolated static func nonverbalPhrase(_ sound: String, plural: Bool) throws -> String {
        guard nonverbalSounds.contains(sound) else {
            throw PackageError.unsupportedRequestFeature("nonverbal sound '\(sound)' — declared: \(nonverbalSounds.joined(separator: ", "))")
        }
        return plural && !sound.hasSuffix("ing") ? sound + "s" : sound
    }

    /// The template language for an edit: Mandarin when the words it carries are (the cookbook gives a CN template for
    /// the content and timbre edits), else English — AuK's trained phrasings in either language. To-whisper always takes
    /// the CN template (the cookbook's own example): the EN one leaves the spectrum where it was on an EN take and a zh
    /// one alike, on both variants (AB-T-0416, MEASUREMENTS "Cookbook").
    nonisolated static func language(for op: SpeechEditOperation) -> AukLanguage {
        if case .style("whisper") = op { return .zh }
        let words: [String]
        switch op {
        case .replaceWords(let original, let new): words = [original, new]
        case .insertWords(let text, let anchor, _): words = [text, anchor]
        case .removeWords(let text, let anchor, _): words = [text, anchor ?? ""]
        case .timbre(let description): words = [description]
        default: words = []
        }
        return words.contains { $0.unicodeScalars.contains(where: isCJK) } ? .zh : .en
    }

    /// The edited take's length in seconds, or nil for the source's own length (upstream's default when `gen_seconds`
    /// is unset). A `targetDuration` wins. Otherwise a speed edit scales the source by 1 / factor (COOKBOOK §3.2), and a
    /// content edit takes upstream's `get_gen_duration` byte ratio when the transcript is known, else adds or takes away
    /// the speaking-rate estimate of the changed words. The nonverbal edits keep the source length: pass a
    /// `targetDuration` to fit a cue.
    nonisolated static func editSeconds(for req: SpeechEditRequest, sourceSeconds: Double) -> Double? {
        if let seconds = req.targetDuration { return seconds }
        let before = req.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        let after = editedTranscript(for: req.edit, transcript: before)
        let byteRatio = before.isEmpty || after.isEmpty ? nil
            : AukInstruction.estimatedSeconds(referenceSeconds: sourceSeconds, referenceText: before, targetText: after)
        switch req.edit {
        case .speed(let factor):
            return sourceSeconds / factor
        case .replaceWords(let original, let new):
            return byteRatio ?? max(0.5, sourceSeconds + speechSeconds(new) - speechSeconds(original))
        case .insertWords(let text, _, _):
            return byteRatio ?? sourceSeconds + speechSeconds(text)
        case .removeWords(let text, _, _):
            return byteRatio ?? max(0.5, sourceSeconds - speechSeconds(text))
        default:
            return nil
        }
    }

    /// The text the edited take should carry (`SpeechEditResponse.transcript`): the request's transcript, with the
    /// words changed for a content edit — or "" when there is no transcript, or a content edit's words are not in it.
    nonisolated static func editedTranscript(for op: SpeechEditOperation, transcript: String) -> String {
        let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "" }
        func spaced(_ words: String) -> String {
            words.unicodeScalars.contains(where: isCJK) ? "" : " "
        }
        func tidy(_ s: String) -> String {
            s.split(separator: " ", omittingEmptySubsequences: true).joined(separator: " ")
                .replacingOccurrences(of: " ,", with: ",").replacingOccurrences(of: " .", with: ".")
        }
        switch op {
        case .replaceWords(let original, let new):
            guard let r = text.range(of: original) else { return "" }
            return text.replacingCharacters(in: r, with: new)
        case .insertWords(let words, let anchor, let before):
            guard let r = text.range(of: anchor) else { return "" }
            let gap = spaced(words + anchor)
            return before ? text.replacingCharacters(in: r, with: words + gap + anchor)
                          : text.replacingCharacters(in: r, with: anchor + gap + words)
        case .removeWords(let words, let anchor, let before):
            var r: Range<String.Index>?
            if let anchor {
                guard let a = text.range(of: anchor) else { return "" }
                r = before ? text.range(of: words, options: .backwards, range: text.startIndex ..< a.lowerBound)
                           : text.range(of: words, range: a.upperBound ..< text.endIndex)
            } else {
                r = text.range(of: words)
            }
            guard let r else { return "" }
            return tidy(text.replacingCharacters(in: r, with: ""))
        default:
            return text
        }
    }

    /// The instruction, audio and length for a TTS request.
    nonisolated static func plan(_ req: TTSRequest) throws -> (AukInstruction, needsAudio: Audio?, seconds: Double?) {
        let text = req.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw PackageError.unsupportedRequestFeature("empty text") }
        guard (req.additionalSpeakers ?? []).isEmpty else {
            throw PackageError.unsupportedRequestFeature("one speaker per request")
        }
        var description: String? = nil
        if let e = req.emotion {
            guard case .textDescription(let d) = e else {
                throw PackageError.unsupportedRequestFeature("emotion mode not declared — AuK takes a textDescription (instruct TTS)")
            }
            description = d
        }
        switch req.voice.selection {
        case .referenceAudio(let audio):
            guard description == nil else {
                throw PackageError.unsupportedRequestFeature("a voice description designs a new voice; it does not combine with a reference clip")
            }
            return (.zeroShotTTS(text: text), audio, req.targetDuration)
        case .auto:
            guard let d = description else {
                throw PackageError.unsupportedRequestFeature("AuK has no built-in voices: pass voice.referenceAudio or emotion.textDescription")
            }
            return (.instructTTS(voiceDescription: d, text: text), nil, req.targetDuration ?? estimatedSpeechSeconds(text))
        case .named:
            throw PackageError.unsupportedRequestFeature("AuK has no named voices")
        @unknown default:
            throw PackageError.unsupportedRequestFeature("voice selection not known to this package")
        }
    }
}

/// Split footprints MEASURED as phys_footprint THROUGH THE ENGINE (`auk-gates validate`: MLXServeEngine(policy: .any)
/// register → prepare (warm-up incl.) → 2 TTS + 3 edits → evict → a second residency cycle; M5 Max, mlx-swift 0.32.3,
/// 2026-10-18; receipts WIP/auk-port/receipts/validate.log). Resident after prepare — Base / Flash: bf16 12.75 / 12.76,
/// int8 9.46 / 9.49, int4 7.63 / 7.90 GB; activation (highest phys over the runs − resident) 0.80–1.03 GB; a 27 s edit
/// leaves the MLX peak where 6 s ones put it (12.60 vs 12.65 GB). Declared with headroom. The second residency cycle
/// re-peaks at the first cycle's level (no leak); phys after evict stays 2.5–9.6 GB with MLX active and cache at 0 —
/// process memory the next load reuses.
///
/// Re-measured 2026-11-01 (AB-T-0414, receipts/staged-residency.txt), the same lane with a 5 ms phys sampler and the
/// kernel's lifetime peak: the IN-request peak is 1.9–2.6 GB over resident (an after-request reading misses it), and
/// Base int4 resident is 7.97 GB. Worst totals: bf16 14.92, int8 11.75, int4 10.04 GB (Base). Activation 1.5 → 2.8 GB
/// and Base int4 7.8 → 8.1 GB, so resident + activation covers every measured peak with ≥ 0.85 GB headroom.
enum AuKFootprints {
    struct Tier { let bf16Resident: UInt64; let int8Resident: UInt64; let int4Resident: UInt64; let peakActivation: UInt64 }
    static let base = Tier(bf16Resident: 12_900_000_000, int8Resident: 9_600_000_000, int4Resident: 8_100_000_000, peakActivation: 2_800_000_000)
    static let flash = Tier(bf16Resident: 12_900_000_000, int8Resident: 9_600_000_000, int4Resident: 8_000_000_000, peakActivation: 2_800_000_000)

    /// `.staged` (AB-T-0414), declared through `FootprintConfigured` as agreed in AB-A-0246: a persistent floor plus a
    /// per-request transient that holds the encoder or the DiT, whichever stage is larger. Measured through the engine
    /// at its default 2 GB MLX pool cap (receipts/staged-residency.txt, both variants): idle after prepare 2.28–2.82 GB;
    /// peak bf16 10.79, int8 7.68, int4 7.34 GB (Base; Flash 6.39). Floor 3.0 GB; transient = peak − 3.0, rounded up.
    /// The read is encoder.safetensors + dit.safetensors, re-read on every request. fp32 is unmeasured: nil.
    struct Staged { let idle: UInt64; let peakTransient: UInt64; let readPerRequest: UInt64 }
    static func staged(_ quant: Quant) -> Staged? {
        switch quant {
        case .bf16: return Staged(idle: 3_000_000_000, peakTransient: 8_000_000_000, readPerRequest: 10_510_000_000)
        case .int8: return Staged(idle: 3_000_000_000, peakTransient: 5_000_000_000, readPerRequest: 7_320_000_000)
        case .int4: return Staged(idle: 3_000_000_000, peakTransient: 4_600_000_000, readPerRequest: 5_620_000_000)
        default: return nil
        }
    }
}

// MARK: - the two registrable packages

@InferenceActor
public final class AuKPackage: ModelPackage {
    public typealias Configuration = AuKConfiguration
    public nonisolated static var manifest: PackageManifest { AuKSurfaces.manifest(variant: .base) }
    let session: AuKSession
    public nonisolated init(configuration: Configuration) {
        var c = configuration; c.variant = .base
        session = AuKSession(configuration: c)
    }
    public func load() async throws { try await session.load() }
    public func unload() async { session.unload() }
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try await AuKRun.run(request, session: session)
    }
    public nonisolated static var registration: PackageRegistration { .of(AuKPackage.self) }
}

@InferenceActor
public final class AuKFlashPackage: ModelPackage {
    public typealias Configuration = AuKConfiguration
    public nonisolated static var manifest: PackageManifest { AuKSurfaces.manifest(variant: .flash) }
    let session: AuKSession
    public nonisolated init(configuration: Configuration) {
        var c = configuration; c.variant = .flash
        session = AuKSession(configuration: c)
    }
    public func load() async throws { try await session.load() }
    public func unload() async { session.unload() }
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try await AuKRun.run(request, session: session)
    }
    public nonisolated static var registration: PackageRegistration { .of(AuKFlashPackage.self) }
}

@InferenceActor
enum AuKRun {
    static func run(_ request: any CapabilityRequest, session: AuKSession) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run().
        try Task.checkCancellation()
        guard session.pipeline != nil else { throw PackageError.notLoaded }
        switch request.capability {
        case .tts:
            guard let req = request as? TTSRequest else { throw PackageError.unsupportedCapability(request.capability) }
            let (instruction, refAudio, seconds) = try AuKSurfaces.plan(req)
            var audio: (samples: [Float], rate: Int)? = nil
            var duration = seconds
            if let refAudio {
                let decoded = try AuKAudioIO.decode(refAudio, what: "reference")
                try checkLength(decoded, what: "reference")
                audio = (decoded.samples, decoded.sampleRate)
                if duration == nil {
                    let refSeconds = Double(decoded.samples.count) / Double(decoded.sampleRate)
                    guard let t = req.referenceTranscript, !t.isEmpty else {
                        throw PackageError.unsupportedRequestFeature("zero-shot TTS needs targetDuration or referenceTranscript (AuK takes an explicit output length)")
                    }
                    duration = AukInstruction.estimatedSeconds(referenceSeconds: refSeconds, referenceText: t, targetText: req.text)
                }
            }
            guard let d = duration, d > 0, d <= AuKSurfaces.maxInputSeconds else {
                throw PackageError.unsupportedRequestFeature("target duration must be in (0, \(Int(AuKSurfaces.maxInputSeconds))] s")
            }
            let seed = req.metaData.intValue("seed").map { UInt64(bitPattern: Int64($0)) } ?? UInt64.random(in: 0 ... UInt64.max)
            try Task.checkCancellation()
            let wav = try session.render(instruction, audio: audio, seconds: d, seed: seed, meta: req.metaData)
            try Task.checkCancellation()
            return TTSResponse(audio: Audio(format: .wav, data: wav, sampleRate: 24000, channels: 1))
        case .speechEdit:
            guard let req = request as? SpeechEditRequest else { throw PackageError.unsupportedCapability(request.capability) }
            // the engine's own pre-flight (contract 1.59.0), so a directly driven package refuses identically
            if let why = AuKSurfaces.speechEditControls().refusal(for: req) {
                throw PackageError.unsupportedRequestFeature(why)
            }
            let instruction = try AuKSurfaces.instruction(for: req.edit)
            let decoded = try AuKAudioIO.decode(req.audio)
            try checkLength(decoded, what: "take")
            let sourceSeconds = Double(decoded.samples.count) / Double(decoded.sampleRate)
            let seconds = AuKSurfaces.editSeconds(for: req, sourceSeconds: sourceSeconds)
            if let s = seconds, s > AuKSurfaces.maxInputSeconds {
                throw PackageError.unsupportedRequestFeature(String(
                    format: "the edited take would be %.1f s — longer than the %.0f s AuK declares; split the take",
                    s, AuKSurfaces.maxInputSeconds))
            }
            let seed = req.seed ?? UInt64.random(in: 0 ... UInt64.max)
            try Task.checkCancellation()
            let wav = try session.render(instruction, language: AuKSurfaces.language(for: req.edit),
                                         audio: (decoded.samples, decoded.sampleRate), seconds: seconds, seed: seed,
                                         meta: req.metaData)
            try Task.checkCancellation()
            return SpeechEditResponse(audio: Audio(format: .wav, data: wav, sampleRate: 24000, channels: 1),
                                      transcript: AuKSurfaces.editedTranscript(for: req.edit, transcript: req.transcript))
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    static func checkLength(_ d: (samples: [Float], sampleRate: Int), what: String) throws {
        let s = Double(d.samples.count) / Double(max(d.sampleRate, 1))
        guard s >= 0.5 else { throw PackageError.unsupportedRequestFeature("\(what) shorter than 0.5 s") }
        guard s <= AuKSurfaces.maxInputSeconds else {
            throw PackageError.unsupportedRequestFeature(String(format: "%@ is %.1f s — longer than the %.0f s AuK declares; split it",
                                                                what, s, AuKSurfaces.maxInputSeconds))
        }
    }
}

public enum AuKPackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    public var description: String {
        switch self { case .missingWeights(let why): return "AuK weights unavailable: \(why)" }
    }
}

extension MetaData {
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }
    func doubleValue(_ key: String) -> Double? {
        switch self[key] {
        case .double(let value)?: return value
        case .int(let value)?: return Double(value)
        default: return nil
        }
    }
}
