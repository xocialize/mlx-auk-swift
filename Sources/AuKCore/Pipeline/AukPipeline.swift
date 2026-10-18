// AukPipeline.swift — auk/infer/infer_auk.py `AukInfer.generate` / `_run`: one instruction (+ optional source /
// reference audio) → 24 kHz audio. Loads the converter's bundles: <model>/{config.json, dit.safetensors,
// vae.safetensors} and <encoder>/{config.json, encoder.safetensors, tokenizer…}.
//
// Upstream behaviour kept:
//   • no audio ⇒ "|<no_prompt_audio>|" is appended to the instruction and the reference is empty;
//   • target length = ceil(gen_seconds · 24000 / 480) latents, or the reference's length when unset;
//   • AuK-Flash ignores caller sampling knobs: 4-step t_grid, CFG off (re-adding CFG clips hard, per upstream);
//   • the reference RMS is measured upstream but never applied, so it is not here either.

import Foundation
import MLX
import MLXNN
import MLXRandom

public struct AukModelConfig: Codable, Sendable {
    public var model_type: String
    public var name: String
    public var is_flash: Bool
    public var arch: Flux2EditConfig
    public var vae: BigVGANFlowVAEConfig
    public var sampling: SamplingConfig
}

public struct AukRequest: Sendable {
    public var instruction: String
    /// Mono source / reference audio and its rate; nil for instruct TTS.
    public var audio: [Float]?
    public var sampleRate: Int
    public var genSeconds: Double?
    public var seed: UInt64
    /// Base AuK only; Flash pins its own recipe.
    public var sampling: SamplingConfig?

    public init(instruction: String, audio: [Float]? = nil, sampleRate: Int = 24000, genSeconds: Double? = nil,
                seed: UInt64 = 0, sampling: SamplingConfig? = nil) {
        self.instruction = instruction; self.audio = audio; self.sampleRate = sampleRate
        self.genSeconds = genSeconds; self.seed = seed; self.sampling = sampling
    }
}

/// Noise a gate injects in place of the pipeline's own draws.
public struct AukInjectedNoise {
    public var y0: MLXArray            // (1, N, D)
    public var vaeNoise: MLXArray?     // (1, Tref, D) channels-last
    public var features: MLXArray?     // (L, 128) encoder features, bypassing the 16 kHz resample + mel
    public init(y0: MLXArray, vaeNoise: MLXArray?, features: MLXArray? = nil) {
        self.y0 = y0; self.vaeNoise = vaeNoise; self.features = features
    }
}

public final class AukPipeline {
    public let config: AukModelConfig
    public let model: CFMEdit
    public let vae: BigVGANFlowVAE
    public let encoder: AukConditionEncoder
    public let dtype: DType

    public init(config: AukModelConfig, model: CFMEdit, vae: BigVGANFlowVAE, encoder: AukConditionEncoder, dtype: DType) {
        self.config = config; self.model = model; self.vae = vae; self.encoder = encoder; self.dtype = dtype
    }

    public static func loadModel(_ dir: URL, dtype: DType, numEncoderLayers: Int = 36) throws -> (AukModelConfig, CFMEdit, BigVGANFlowVAE) {
        let config = try ConfigIO.load(AukModelConfig.self, from: dir.appendingPathComponent("config.json"))
        let model = CFMEdit(config.arch, numEncoderLayers: numEncoderLayers)
        let keep: (String) -> Bool = { $0.hasSuffix("layer_weights") || $0.hasSuffix("layer_scale") || $0.hasSuffix("inv_freq") }
        try WeightIO.apply(try WeightIO.load(dir.appendingPathComponent("dit.safetensors"), dtype: dtype, keepFP32: keep),
                           to: model, component: "dit")
        let vae = BigVGANFlowVAE(config.vae)
        // the VAE stays fp32 below bf16 DiT tiers only if asked; default follows `dtype`, statistics + taps fp32
        let keepVAE: (String) -> Bool = { $0.hasSuffix("global_mean") || $0.hasSuffix("global_log_std") || $0.hasSuffix("filter") }
        try WeightIO.apply(try WeightIO.load(dir.appendingPathComponent("vae.safetensors"), dtype: dtype, keepFP32: keepVAE),
                           to: vae, component: "vae")
        return (config, model, vae)
    }

    public static func load(modelDir: URL, encoderDir: URL, dtype: DType, encoderDType: DType? = nil) async throws -> AukPipeline {
        let (config, model, vae) = try loadModel(modelDir, dtype: dtype)
        let encoder = try await QwenOmniConditionEncoder(directory: encoderDir, dtype: encoderDType ?? dtype)
        return AukPipeline(config: config, model: model, vae: vae, encoder: encoder, dtype: dtype)
    }

    public var sampleRate: Int { config.vae.target_sample_rate }
    var hop: Int { config.vae.downsample_rate }

    /// Runs one request. `inject` replaces the pipeline's random draws (gates).
    public func generate(_ req: AukRequest, inject: AukInjectedNoise? = nil,
                         onStep: ((Int, Int) -> Void)? = nil,
                         capture: ((String, MLXArray) -> Void)? = nil) throws -> [Float] {
        var instruction = req.instruction
        var refLatents = MLXArray.zeros([1, 0, config.vae.latent_dim])
        var audio16k: [Float]? = nil
        var refLatentLen = 0

        if let audio = req.audio, !audio.isEmpty {
            let ref24 = Resample.sinc(audio, from: req.sampleRate, to: sampleRate)
            refLatentLen = ref24.count / hop
            audio16k = Resample.sinc(audio, from: req.sampleRate, to: 16000)
            let x = MLXArray(ref24).reshaped(1, -1, 1).asType(dtype)
            let (mean, _) = vae.encodeStats(x)
            let noise = inject?.vaeNoise ?? MLXRandom.normal(mean.shape, key: MLXRandom.key(req.seed &+ 1))
            let lat = vae.encodingAndNormalization(x, noise: noise)
            refLatentLen = min(refLatentLen, lat.dim(1))   // lat may hold one frame more; it stays, masked
            eval(lat)
            refLatents = lat
            capture?("ref_latents", lat)
        } else if !instruction.hasSuffix("|<no_prompt_audio>|") {
            instruction += "|<no_prompt_audio>|"
        }

        let genLen: Int
        if let s = req.genSeconds {
            genLen = max(1, Int(ceil(s * Double(sampleRate) / Double(hop))))
        } else {
            genLen = max(1, refLatentLen)
        }

        let prompt = AukPrompt(instruction: instruction, audio16k: audio16k)
        let (textEmbeds, contextMask): (MLXArray, MLXArray)
        if let enc = encoder as? QwenOmniConditionEncoder, let feats = inject?.features {
            let w = softmax(model.layerWeights.asType(.float32), axis: 0)
            var acc: MLXArray? = nil
            let n = try enc.forEachHiddenState(prompt, featuresOverride: feats) { i, h in
                guard i >= 1 else { return }
                let term = MLXFast.layerNorm(h.asType(.float32), weight: nil, bias: nil, eps: 1e-5) * w[i - 1]
                acc = acc.map { $0 + term } ?? term
                eval(acc!)
            }
            textEmbeds = acc! * model.layerScale.asType(.float32)
            contextMask = MLXArray.ones([1, n]).asType(.bool)
        } else {
            (textEmbeds, contextMask) = try model.encodeText(prompt, encoder: encoder)
        }
        eval(textEmbeds)
        capture?("fused", textEmbeds)

        let sampling = config.is_flash ? config.sampling : (req.sampling ?? config.sampling)
        let y0 = inject?.y0 ?? MLXRandom.normal([1, genLen, config.vae.latent_dim], key: MLXRandom.key(req.seed))
        precondition(y0.dim(1) == genLen, "injected y0 has \(y0.dim(1)) frames, request needs \(genLen)")
        let gen = model.sample(cond: refLatents, refLen: refLatentLen, textEmbeds: textEmbeds, contextMask: contextMask, y0: y0,
                               sampling: sampling, dtype: dtype, onStep: onStep, capture: capture)
        capture?("gen_latent", gen)

        let wave = vae.inferenceFromLatents(vae.denormalize(gen).asType(dtype))
        eval(wave)
        capture?("audio_out", wave)
        return wave.asType(.float32).reshaped(-1).asArray(Float.self)
    }
}
