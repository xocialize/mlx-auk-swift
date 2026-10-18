// AukInstruction.swift — AuK's canonical instruction templates (docs/COOKBOOK.md @ 9481474), typed. The model was
// trained on these phrasings; the Prompt Enhancer (pe.py, an external LLM + ASR) exists to rewrite free-form requests
// into them, so a typed builder is the on-device substitute. English templates; Mandarin variants where the cookbook
// gives one.

import Foundation

public enum AukLanguage: String, Sendable, Codable { case en, zh }

public enum AukInstruction: Sendable, Equatable {
    // 1. generation
    case zeroShotTTS(text: String)                                   // needs a reference clip
    case instructTTS(voiceDescription: String, text: String)         // no audio
    // 2. content
    case replace(original: String, new: String)
    case insert(text: String, anchor: String, before: Bool)
    case remove(text: String, anchor: String?, before: Bool)
    case lyricEdit(original: String, new: String)
    // 3. acoustic
    case pitch(semitones: Int)                                       // ±1…3
    case speed(factor: Double)                                       // 0.5, 0.75, 1.25, 1.5, 2.0
    case volume(dB: Int)                                             // ±5, 10, 15
    // 4. paralinguistic
    case emotion(String)                                             // happy, angry, sad, fearful, surprised, disgusted, calm, excited
    case timbre(description: String)
    case deAccent
    case removeNonverbal(String)                                     // "breaths", "laughs", "coughs", "humming" …
    case addNonverbal(sound: String, atEnd: Bool)
    case toWhisper
    case fromWhisper
    // 5. enhancement / separation
    case denoise
    case dereverberate
    case enhance
    case keepSpeaker(ordinal: Int)                                   // 1 = first to start talking
    case keepSpeakerSaying(String)
    case singingOnly
    case allVoices
    /// An instruction already in canonical form (escape hatch for the cookbook's long tail).
    case raw(String)

    public static let emotions = ["happy", "angry", "sad", "fearful", "surprised", "disgusted", "calm", "excited"]
    public static let speedFactors: [Double] = [0.5, 0.75, 1.25, 1.5, 2.0]
    public static let pitchSteps = [-3, -2, -1, 1, 2, 3]
    public static let volumeSteps = [-15, -10, -5, 5, 10, 15]

    /// Whether the task edits / conditions on a source clip.
    public var needsAudio: Bool {
        switch self { case .instructTTS: return false; case .raw: return true; default: return true }
    }

    /// Whether the output length follows the source (upstream's default when gen_seconds is unset) — otherwise the
    /// caller supplies a duration.
    public var keepsSourceLength: Bool {
        switch self {
        case .zeroShotTTS, .instructTTS, .replace, .insert, .remove, .lyricEdit, .speed, .addNonverbal, .removeNonverbal:
            return false
        default:
            return true
        }
    }

    private static func ordinal(_ n: Int) -> String {
        let words = ["first", "second", "third", "fourth", "fifth"]
        return n >= 1 && n <= words.count ? words[n - 1] : "\(n)th"
    }

    private static func num(_ x: Double) -> String {
        x == x.rounded() ? String(format: "%.1f", x) : String(x)
    }

    public func text(_ lang: AukLanguage = .en) -> String {
        switch (self, lang) {
        case (.zeroShotTTS(let t), _): return "Say the following with the same voice: \"\(t)\""
        case (.instructTTS(let d, let t), .en): return "Generate speech based on the following description: \"\(d)\". The content to speak is: \"\(t)\"."
        case (.instructTTS(let d, let t), .zh): return "请基于下面的描述: \"\(d)\",生成语音内容\"\(t)\"."
        case (.replace(let o, let n), .en): return "Replace '\(o)' with '\(n)'."
        case (.replace(let o, let n), .zh): return "把‘\(o)’改成‘\(n)’"
        case (.insert(let t, let a, let before), .en): return "Add '\(t)' \(before ? "before" : "after") '\(a)'."
        case (.insert(let t, let a, let before), .zh): return "在‘\(a)’\(before ? "前面" : "后面")加上‘\(t)’"
        case (.remove(let t, nil, _), _): return "Remove '\(t)'."
        case (.remove(let t, let a?, let before), _): return "Remove '\(t)' \(before ? "before" : "after") '\(a)'."
        case (.lyricEdit(let o, let n), .en): return "Change \"\(o)\" to \"\(n)\" in the vocal recording."
        case (.lyricEdit(let o, let n), .zh): return "把这段歌词中的“\(o)”改成“\(n)”。"
        case (.pitch(let s), .en): return s >= 0 ? "Raise the pitch by \(s) semitones." : "Lower the pitch by \(-s) semitones."
        case (.pitch(let s), .zh): return s >= 0 ? "将音调升高\(s)个半音。" : "将音调降低\(-s)个半音。"
        case (.speed(let f), .en): return "Adjust the speech speed to \(Self.num(f))x."
        case (.speed(let f), .zh): return "将语速调整为\(Self.num(f))倍。"
        case (.volume(let d), .en): return d >= 0 ? "Increase the volume by \(d) dB." : "Decrease the volume by \(-d) dB."
        case (.volume(let d), .zh): return d >= 0 ? "将音量升高\(d)分贝。" : "将音量降低\(-d)分贝。"
        case (.emotion(let e), _): return "Change the emotion to \(e)."
        case (.timbre(let d), .en): return "Keep the spoken content unchanged and change the timbre to: \"\(d)\"."
        case (.timbre(let d), .zh): return "请将这段音频的音色修改为符合以下描述的声音：“\(d)”。"
        case (.deAccent, .en): return "Remove the regional accent while preserving the speaker's voice and content."
        case (.deAccent, .zh): return "请去掉这段语音里的方言口音，保持说话人音色一致。"
        case (.removeNonverbal(let s), _): return "Remove all \(s) from the audio."
        case (.addNonverbal(let s, let end), _): return "Add a \(s) at the \(end ? "end" : "beginning") of the speech."
        case (.toWhisper, .en): return "Convert this speech into a soft whisper while preserving the speaker and content."
        case (.toWhisper, .zh): return "用小声耳语的方式把这段话说出来。"
        case (.fromWhisper, .en): return "Convert this whispered speech into a normal speaking voice while preserving the speaker and content."
        case (.fromWhisper, .zh): return "把这段耳语转换成正常说话的声音。"
        case (.denoise, .en): return "Remove only the background noise, preserve everything else, and output audio of the same length."
        case (.denoise, .zh): return "请只去除背景噪声，保留其他内容，输出等长结果。"
        case (.dereverberate, .en): return "Remove only the room reverberation, preserve everything else, and output audio of the same length."
        case (.dereverberate, .zh): return "请只去除房间混响，保留其他内容，输出等长结果。"
        case (.enhance, _): return "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length."
        case (.keepSpeaker(let n), .en): return "Keep only the \(Self.ordinal(n)) speaker to start talking and remove all other speakers."
        case (.keepSpeaker(let n), .zh): return "只保留第\(n)个开始说话的人，去掉其余说话人。"
        case (.keepSpeakerSaying(let c), .en): return "Keep only the speaker who says \"\(c)\" and remove all other speakers."
        case (.keepSpeakerSaying(let c), .zh): return "请只保留说“\(c)”的人，去掉其他说话人。"
        case (.singingOnly, .en): return "Keep only the singing voice and remove everything else."
        case (.singingOnly, .zh): return "请只保留歌声，其余声音都去掉。"
        case (.allVoices, .en): return "Keep all human voices, including speech and singing, and remove everything else."
        case (.allVoices, .zh): return "请保留所有人声，包括说话和歌唱，其余声音都去掉。"
        case (.raw(let s), _): return s
        default: return text(.en)
        }
    }

    /// upstream `get_gen_duration`: reference seconds × bytes(gen_text) / bytes(ref_text) / speed, when both texts are
    /// known; else the source length (nil here).
    public static func estimatedSeconds(referenceSeconds: Double, referenceText: String, targetText: String,
                                        speed: Double = 1.0) -> Double {
        referenceSeconds * Double(targetText.utf8.count) / Double(max(1, referenceText.utf8.count)) / speed
    }
}
