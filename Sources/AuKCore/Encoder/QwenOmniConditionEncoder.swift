// QwenOmniConditionEncoder.swift — AuK's semantic encoder as released: the Qwen2.5-Omni-3B thinker behind
// `Qwen2_5OmniProcessor` (ChatML + WhisperFeatureExtractor). This is the Qwen RESEARCH LICENSE component
// (non-commercial); the DiT and VAE are MIT. See AB-D-0153.
//
// Processor, as transformers 4.57.1 runs it for AuK's single user turn [text, audio]:
//   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}
//   <|audio_bos|><|AUDIO|>×N<|audio_eos|><|im_end|>\n<|im_start|>assistant\n
// N = ((L − 1) // 2 + 1 − 2) // 2 + 1 for L valid mel frames. Features: 16 kHz, n_fft 400, hop 160, 128 Slaney mels,
// centred reflect-padded STFT over the zero-padded clip, |·|², drop the last frame, log10 (floor 1e-10),
// max − 8 clamp, (x + 4) / 4. Upstream resamples to 16 kHz with librosa (soxr_hq); we use the torchaudio sinc kernel
// and gate the difference on the features.

import Foundation
import MLX
import MLXNN
import Tokenizers

/// One AuK request as the encoder sees it.
public struct AukPrompt: Sendable {
    public var instruction: String
    /// Mono 16 kHz audio for the encoder, or nil for text-only (instruct TTS).
    public var audio16k: [Float]?
    public var systemPrompt: String

    public init(instruction: String, audio16k: [Float]?, systemPrompt: String = "You are a helpful assistant.") {
        self.instruction = instruction
        self.audio16k = audio16k
        self.systemPrompt = systemPrompt
    }
}

public final class QwenOmniConditionEncoder: AukConditionEncoder {
    public let thinker: Qwen2_5OmniThinker
    let tokenizer: Tokenizer
    let melFilters: MLXArray
    public var numLayers: Int { thinker.config.text_config.num_hidden_layers }
    public let weightLicense = "LicenseRef-Qwen-Research"
    public let dtype: DType

    /// Last prompt's processor outputs, for the gates.
    public private(set) var lastInputIds: [Int32] = []
    public private(set) var lastFeatures: MLXArray?
    public private(set) var lastAudioEmbeds: MLXArray?

    public init(directory: URL, dtype: DType) async throws {
        let config = try ConfigIO.load(QwenOmniThinkerConfig.self, from: directory.appendingPathComponent("config.json"))
        let thinker = Qwen2_5OmniThinker(config)
        let weights = try WeightIO.load(directory.appendingPathComponent("encoder.safetensors"), dtype: dtype)
        try WeightIO.apply(WeightIO.scoped(weights, prefix: "thinker."), to: thinker, component: "thinker")
        self.thinker = thinker
        self.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        self.melFilters = Signal.slaneyMelFilters(sampleRate: 16000, nFFT: 400, nMels: config.audio_config.num_mel_bins)
        self.dtype = dtype
    }

    /// WhisperFeatureExtractor on one clip → (L_valid, 128) features, L_valid = ceil(n / 160).
    public func features(_ audio16k: [Float]) -> MLXArray {
        let nValid = (audio16k.count + 159) / 160
        let padded = audio16k + [Float](repeating: 0, count: 400 + 160)
        let logMel = Signal.whisperLogMel(MLXArray(padded), filters: melFilters, nFFT: 400, hop: 160)  // (128, T)
        return logMel[0..., ..<nValid].transposed(1, 0)
    }

    /// The token ids the processor produces for `prompt` with `audioTokens` placeholders.
    public func inputIds(_ prompt: AukPrompt, audioTokens: Int?) -> [Int32] {
        let head = "<|im_start|>system\n\(prompt.systemPrompt)<|im_end|>\n<|im_start|>user\n\(prompt.instruction)"
        let tail = "<|im_end|>\n<|im_start|>assistant\n"
        var ids = tokenizer.encode(text: head, addSpecialTokens: false).map(Int32.init)
        if let n = audioTokens {
            ids.append(Int32(thinker.config.audio_start_token_id))
            ids += [Int32](repeating: Int32(thinker.config.audio_token_index), count: n)
            ids.append(Int32(thinker.config.audio_end_token_id))
        }
        ids += tokenizer.encode(text: tail, addSpecialTokens: false).map(Int32.init)
        return ids
    }

    /// Runs the audio tower on `features` (L, 128), or reuses precomputed features from a gate.
    public func audioEmbeds(features: MLXArray) -> MLXArray {
        thinker.audioTower(features.asType(dtype))
    }

    public func forEachHiddenState(_ prompt: AukPrompt, visit: (Int, MLXArray) -> Void) throws -> Int {
        try forEachHiddenState(prompt, featuresOverride: nil, visit: visit)
    }

    public func forEachHiddenState(_ prompt: AukPrompt, featuresOverride: MLXArray?,
                                   visit: (Int, MLXArray) -> Void) throws -> Int {
        var embedsForAudio: MLXArray? = nil
        var nTokens: Int? = nil
        if let feats = featuresOverride ?? prompt.audio16k.map({ features($0) }) {
            lastFeatures = feats
            let a = audioEmbeds(features: feats)
            eval(a)
            embedsForAudio = a
            nTokens = a.dim(0)
            let expected = Qwen2_5OmniAudioEncoder.outputLength(melFrames: feats.dim(0))
            guard expected == a.dim(0) else {
                throw AuKError.invalidInput("audio tower gave \(a.dim(0)) tokens, processor reserves \(expected)")
            }
        }
        lastAudioEmbeds = embedsForAudio
        let ids = inputIds(prompt, audioTokens: nTokens)
        lastInputIds = ids
        let embeds = thinker.inputsEmbeds(inputIds: ids, audioEmbeds: embedsForAudio)
        thinker.model.forEachHiddenState(embeds, visit: visit)
        return ids.count
    }
}
