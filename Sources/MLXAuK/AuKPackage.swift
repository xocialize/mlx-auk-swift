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
/// `speechEdit`: `.emotion` (AuK's 8 labels), `.style("whisper")` / `.style("normal")` (whisper conversion both ways),
/// `.denoise`. The output keeps the source's length. The transcript is not needed by AuK and is returned unchanged.
/// AuK's content, speed, pitch, volume, timbre and de-accent edits wait on the contract (AB-A-0188).
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
        pipeline = try await AukPipeline.load(modelDir: modelDir, encoderDir: encDir, dtype: configuration.dtype)
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

    func render(_ instruction: AukInstruction, audio: (samples: [Float], rate: Int)?, seconds: Double?, seed: UInt64,
                meta: MetaData) throws -> Data {
        guard let pipeline else { throw PackageError.notLoaded }
        var sampling: SamplingConfig? = nil
        if !pipeline.config.is_flash, meta.intValue("nfe") != nil || meta.doubleValue("cfgStrength") != nil {
            var s = pipeline.config.sampling
            if let n = meta.intValue("nfe") { s.nfe = n }
            if let c = meta.doubleValue("cfgStrength") { s.cfg_strength = c }
            sampling = s
        }
        let wave = try pipeline.generate(
            AukRequest(instruction: instruction.text(), audio: audio?.samples, sampleRate: audio?.rate ?? 24000,
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
    /// Instruct-TTS length estimate when no targetDuration is given (Latin ≈ 14 chars/s, CJK ≈ 4.5 chars/s).
    nonisolated static func estimatedSpeechSeconds(_ text: String) -> Double {
        var cjk = 0, other = 0
        for s in text.unicodeScalars {
            if (0x4E00 ... 0x9FFF).contains(s.value) || (0x3400 ... 0x4DBF).contains(s.value) { cjk += 1 } else { other += 1 }
        }
        return max(0.8, Double(cjk) / 4.5 + Double(other) / 14.0)
    }

    nonisolated static func speechEditControls() -> SpeechEditControls {
        SpeechEditControls(operations: [.emotion, .style, .denoise], emotionLabels: emotionLabels, styleLabels: styleLabels,
                           paralinguisticTags: [], maxInputSeconds: maxInputSeconds)
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
                        + "an existing take re-delivered with an emotion (happy, angry, sad, fearful, surprised, disgusted, calm, "
                        + "excited), converted to or from a whisper, or denoised — same voice, same words, same length. ≤ 30 s. "
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
        case .paralinguistic, .trimSilence:
            throw PackageError.unsupportedRequestFeature("speech edit kind not declared by AuK")
        @unknown default:
            throw PackageError.unsupportedRequestFeature("speech edit kind not known to this package")
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
enum AuKFootprints {
    struct Tier { let bf16Resident: UInt64; let int8Resident: UInt64; let int4Resident: UInt64; let peakActivation: UInt64 }
    static let base = Tier(bf16Resident: 12_900_000_000, int8Resident: 9_600_000_000, int4Resident: 7_800_000_000, peakActivation: 1_500_000_000)
    static let flash = Tier(bf16Resident: 12_900_000_000, int8Resident: 9_600_000_000, int4Resident: 8_000_000_000, peakActivation: 1_500_000_000)
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
            let instruction = try AuKSurfaces.instruction(for: req.edit)
            let decoded = try AuKAudioIO.decode(req.audio)
            try checkLength(decoded, what: "take")
            let seed = req.seed ?? UInt64.random(in: 0 ... UInt64.max)
            try Task.checkCancellation()
            let wav = try session.render(instruction, audio: (decoded.samples, decoded.sampleRate), seconds: nil, seed: seed,
                                         meta: req.metaData)
            try Task.checkCancellation()
            return SpeechEditResponse(audio: Audio(format: .wav, data: wav, sampleRate: 24000, channels: 1),
                                      transcript: req.transcript)
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
