// auk-gates cookbook — every docs/COOKBOOK.md example MLXAuK's surfaces can express, through MLXServeEngine (AB-T-0416).
//
//   auk-gates cookbook <base|flash> <bf16|int8|int4> <model> <encoder> <assets> <out-dir> [seed]
//
// The case list follows upstream's scripts/run_cookbook_mlx.py (Tencent-Hunyuan/AuK feat/mlx-apple-silicon @ 6943a1e),
// one row per cookbook example, with the request each example becomes on the `tts` / `speechEdit` surfaces. Examples
// no surface can express are written to the manifest as `unexpressible` with the reason and are not run. Rows marked
// `supplementary` cover declared edit kinds the cookbook has no example for; Tools/cookbook/verify_cookbook.py reports
// them apart from the cookbook count. Writes <out-dir>/<id>.wav and <out-dir>/cases.jsonl.

import AuKCore
import Foundation
import MLXAuK
import MLXServeCore
import MLXToolKit

struct CookbookCase {
    let id: String
    let section: String
    let kind: String              // cookbook | supplementary | unexpressible | excluded
    let request: String           // the request as the surface sees it (for the manifest)
    let note: String
    let make: (() throws -> any CapabilityRequest)?
}

func cookbook(variant: String, quant q: String, _ modelDir: URL, _ encDir: URL, _ assets: URL, _ outDir: URL,
              seed: UInt64) async throws {
    let quant: Quant = q == "int8" ? .int8 : q == "int4" ? .int4 : .bf16
    let v: AuKVariant = variant == "base" ? .base : .flash
    let config = AuKConfiguration(variant: v, quant: quant, modelDirectory: modelDir, encoderDirectory: encDir)
    let registration: PackageRegistration = v == .base ? AuKPackage.registration : AuKFlashPackage.registration
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    func wav(_ rel: String) throws -> Audio {
        Audio(format: .wav, data: try Data(contentsOf: assets.appendingPathComponent(rel)))
    }
    let tts: Int64 = Int64(bitPattern: seed)
    let itts = "一位二十多岁的女性，面对刚到家的伴侣，以温柔、关心且略带撒娇的语气轻声诉说。她的声音柔和亲密，语速稍缓，音量适中，"
        + "音色甜美自然，在短语结尾处语调温暖地上扬。她正向伴侣问候回家，并温柔询问今日工作如何。"
    let replaced = (original: "but accepting what we cannot have", new: "and living well with dreams unmet")
    let cases: [CookbookCase] = [
        CookbookCase(id: "1.1-zeroshot-tts", section: "Zero-shot TTS", kind: "cookbook",
                     request: "tts voice=.referenceAudio(zero-shot-tts/ref.wav) targetDuration=6.0", note: "") {
            TTSRequest(text: "Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience",
                       voice: VoiceSelector(.referenceAudio(try wav("zero-shot-tts/ref.wav"))), targetDuration: 6.0,
                       metaData: ["seed": .int(Int(tts))])
        },
        CookbookCase(id: "1.2-instruct-tts", section: "Instruct TTS", kind: "cookbook",
                     request: "tts emotion=.textDescription(<cookbook's Mandarin description>) voice=.auto targetDuration=1.7",
                     note: "the package's EN instruct template, not the cookbook's free phrasing") {
            TTSRequest(text: "Welcome home, how was work today?", emotion: .textDescription(itts), targetDuration: 1.7,
                       metaData: ["seed": .int(Int(tts))])
        },
        CookbookCase(id: "2.1-content-edit", section: "Speech Content Editing", kind: "cookbook",
                     request: "speechEdit .replaceWords(original:new:) targetDuration=7.0", note: "") {
            SpeechEditRequest(audio: try wav("content-edit/content.wav"),
                              edit: .replaceWords(original: replaced.original, new: replaced.new), targetDuration: 7.0, seed: seed)
        },
        CookbookCase(id: "2.2-lyric-edit", section: "Lyric Editing", kind: "excluded",
                     request: "—", note: "known model limit: fails in the torch reference too at the same steps and seed (AB-T-0416)",
                     make: nil),
        CookbookCase(id: "3.1-pitch", section: "Pitch Editing", kind: "cookbook", request: "speechEdit .pitch(semitones: 2)", note: "") {
            SpeechEditRequest(audio: try wav("pitch/pitch-1-input.wav"), edit: .pitch(semitones: 2), seed: seed)
        },
        CookbookCase(id: "3.2-speed", section: "Speed Editing", kind: "cookbook",
                     request: "speechEdit .speed(factor: 1.5) targetDuration=6.86", note: "") {
            SpeechEditRequest(audio: try wav("speed/speed-edit-1-input.wav"), edit: .speed(factor: 1.5), targetDuration: 6.86, seed: seed)
        },
        CookbookCase(id: "3.3-volume", section: "Volume Editing", kind: "cookbook", request: "speechEdit .volume(dB: 10)", note: "") {
            SpeechEditRequest(audio: try wav("energy/energy-edit-1-input.wav"), edit: .volume(dB: 10), seed: seed)
        },
        CookbookCase(id: "4.1-emotion", section: "Emotion", kind: "cookbook", request: "speechEdit .emotion(\"happy\")", note: "") {
            SpeechEditRequest(audio: try wav("emotion-edit/en-1-input.wav"), edit: .emotion("happy"), seed: seed)
        },
        CookbookCase(id: "4.2-timbre", section: "Timbre", kind: "cookbook",
                     request: "speechEdit .timbre(\"a deep, calm male voice\")", note: "") {
            SpeechEditRequest(audio: try wav("vc/vc-1-input.wav"), edit: .timbre(description: "a deep, calm male voice"), seed: seed)
        },
        CookbookCase(id: "4.3-deaccent", section: "De-accent", kind: "cookbook", request: "speechEdit .deAccent",
                     note: "the package's EN de-accent template; the cookbook example is phrased in Mandarin") {
            SpeechEditRequest(audio: try wav("accent/accent-sichuan-input.wav"), edit: .deAccent, seed: seed)
        },
        CookbookCase(id: "4.4-nonverbal-remove", section: "Nonverbal Remove", kind: "cookbook",
                     request: "speechEdit .removeNonverbal(sound: \"humming\") targetDuration=22.0",
                     note: "\"Remove all humming from the audio.\" (the cookbook example says \"the humming\")") {
            SpeechEditRequest(audio: try wav("nv/en-d-input.wav"), edit: .removeNonverbal(sound: "humming"), targetDuration: 22.0, seed: seed)
        },
        CookbookCase(id: "4.4-nonverbal-add", section: "Nonverbal Add", kind: "unexpressible", request: "—",
                     note: "inline position: the cookbook adds a cough BEFORE 'We tested'; .addNonverbal takes only the start or end of the take",
                     make: nil),
        CookbookCase(id: "4.5-whisper", section: "Whisper Conversion", kind: "cookbook", request: "speechEdit .style(\"whisper\")",
                     note: "the package's EN whisper template; the cookbook example is phrased in Mandarin") {
            SpeechEditRequest(audio: try wav("whisper/wh-w2n-zh-input.wav"), edit: .style("whisper"), seed: seed)
        },
        CookbookCase(id: "5.1-enhance", section: "Speech Enhancement", kind: "cookbook", request: "speechEdit .denoise",
                     note: "approximated: the cookbook's combined enhance (noise + reverb) has no single kind; .denoise carries the noise-floor check") {
            SpeechEditRequest(audio: try wav("se/se-zh-1-input.wav"), edit: .denoise, seed: seed)
        },
        CookbookCase(id: "5.2-separation", section: "Speech Separation", kind: "unexpressible", request: "—",
                     note: "no audioSeparation surface (keep the second speaker to start talking)", make: nil),
        CookbookCase(id: "5.3-music-sep", section: "Music Separation", kind: "unexpressible", request: "—",
                     note: "no audioSeparation surface (keep the singing voice)", make: nil),
        CookbookCase(id: "5.4-tse", section: "Target Speaker Extraction", kind: "unexpressible", request: "—",
                     note: "no audioSeparation surface (keep the speaker who says \"get what\")", make: nil),
        // declared kinds the cookbook has no example for, on the cookbook's own inputs
        CookbookCase(id: "S-insert", section: "Content insert", kind: "supplementary",
                     request: "speechEdit .insertWords(\"quietly\", anchor: \"accepting\", before: true)", note: "length from the package's estimate") {
            SpeechEditRequest(audio: try wav("content-edit/content.wav"),
                              edit: .insertWords(text: "quietly", anchor: "accepting", before: true), seed: seed)
        },
        CookbookCase(id: "S-remove", section: "Content remove", kind: "supplementary",
                     request: "speechEdit .removeWords(\"\(replaced.original)\")", note: "length from the package's estimate") {
            SpeechEditRequest(audio: try wav("content-edit/content.wav"), edit: .removeWords(text: replaced.original, anchor: nil, before: false), seed: seed)
        },
        CookbookCase(id: "S-nonverbal-add-start", section: "Nonverbal add (edge)", kind: "supplementary",
                     request: "speechEdit .addNonverbal(sound: \"cough\", atEnd: false) targetDuration=14.3",
                     note: "the expressible neighbour of 4.4-add: a cough at the start, +1 s") {
            SpeechEditRequest(audio: try wav("nv/en-c-input.wav"), edit: .addNonverbal(sound: "cough", atEnd: false),
                              targetDuration: 14.3, seed: seed)
        },
        CookbookCase(id: "S-dereverb", section: "Dereverberate", kind: "supplementary", request: "speechEdit .dereverberate", note: "") {
            SpeechEditRequest(audio: try wav("se/se-zh-1-input.wav"), edit: .dereverberate, seed: seed)
        },
        CookbookCase(id: "S-from-whisper", section: "Whisper → normal", kind: "supplementary", request: "speechEdit .style(\"normal\")",
                     note: "the input is the cookbook's whisper→normal sample") {
            SpeechEditRequest(audio: try wav("whisper/wh-w2n-zh-input.wav"), edit: .style("normal"), seed: seed)
        },
    ]

    let engine = MLXServeEngine(policy: .any)
    let id = try await engine.register(registration, configuration: config, id: PackageID(v == .base ? "auk" : "auk-flash"))
    let t0 = Date()
    try await engine.prepare(.tts, package: id)
    try await engine.prepare(.speechEdit, package: id)
    print(String(format: "# cookbook %@ %@ seed %llu — prepared in %.1fs", variant, q, seed, Date().timeIntervalSince(t0)))

    var manifest = ""
    var total = 0.0
    for c in cases {
        var row: [String: Any] = ["id": c.id, "section": c.section, "kind": c.kind, "request": c.request, "note": c.note,
                                  "variant": variant, "quant": q, "seed": seed]
        if let make = c.make {
            do {
                let req = try make()
                let t = Date()
                let resp = try await engine.run(req, package: id)
                let wall = Date().timeIntervalSince(t); total += wall
                let audio = (resp as? TTSResponse)?.audio ?? (resp as! SpeechEditResponse).audio
                try audio.data.write(to: outDir.appendingPathComponent("\(c.id).wav"))
                let (s, r) = try WAV.read(outDir.appendingPathComponent("\(c.id).wav"))
                let secs = Double(s.count) / Double(r)
                row["status"] = "ok"; row["wall_s"] = wall; row["out_s"] = secs
                print("  ok   " + c.id.padding(toLength: 24, withPad: " ", startingAt: 0) + String(format: "%6.1fs wall  %5.2fs out", wall, secs))
            } catch {
                row["status"] = "FAIL"; row["error"] = String(describing: error)
                print("  FAIL \(c.id): \(String(describing: error).prefix(160))")
            }
        } else {
            row["status"] = c.kind
            print("  --   \(c.id): \(c.kind) — \(c.note)")
        }
        let line = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        manifest += String(decoding: line, as: UTF8.self) + "\n"
    }
    try manifest.write(to: outDir.appendingPathComponent("cases.jsonl"), atomically: true, encoding: .utf8)
    await engine.evict(package: id)
    print(String(format: "# %.0fs total render wall → %@", total, outDir.path))
}
