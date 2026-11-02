// AukPipeline.swift — auk/infer/infer_auk.py `AukInfer.generate` / `_run`: one instruction (+ optional source /
// reference audio) → 24 kHz audio. Loads the converter's bundles: <model>/{config.json, dit.safetensors,
// vae.safetensors} and <encoder>/{config.json, encoder.safetensors, tokenizer…}.
//
// Upstream behaviour kept:
//   • no audio ⇒ "|<no_prompt_audio>|" is appended to the instruction and the reference is empty;
//   • target length = ceil(gen_seconds · 24000 / 480) latents, or the reference's length when unset;
//   • AuK-Flash ignores caller sampling knobs: 4-step t_grid, CFG off (re-adding CFG clips hard, per upstream);
//   • the reference RMS is measured upstream but never applied, so it is not here either.
//
// Residency (AB-T-0414): `.resident` holds encoder + DiT + VAE for the pipeline's lifetime. `.staged` holds only the
// VAE, the fusion head and the tokenizer; each call reads the encoder, conditions, frees it, then reads the DiT,
// samples, frees it — one large model in memory at a time (upstream's `--sequential`, feat/mlx-apple-silicon). The
// arithmetic is the same either way, so the output is bit-identical (`auk-gates staged`).

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

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

/// Runtime quantization tiers (PORTING-SPEC §6): affine, per-group, applied after the bf16 weights load.
public struct AukQuantization: Sendable {
    public var ditBits: Int?
    public var encoderBits: Int?
    public var groupSize: Int

    public static let none = AukQuantization(ditBits: nil, encoderBits: nil, groupSize: 64)

    public init(ditBits: Int?, encoderBits: Int?, groupSize: Int = 64) {
        self.ditBits = ditBits; self.encoderBits = encoderBits; self.groupSize = groupSize
    }

    /// DiT: the Linears inside the 30 transformer blocks (attention, AdaLN modulation, SwiGLU). Encoder: every Linear
    /// inside the 36 LLM layers and the 32 audio-tower layers. Embeddings, projections, norms and the VAE stay as loaded.
    public func apply(dit: CFMEdit, thinker: Qwen2_5OmniThinker) {
        apply(dit: dit)
        apply(thinker: thinker)
    }

    public func apply(dit: CFMEdit) {
        if let b = ditBits {
            quantize(model: dit.transformer, groupSize: groupSize, bits: b) { path, m in
                m is Linear && (path.hasPrefix("transformer_blocks.") || path.hasPrefix("single_transformer_blocks."))
            }
            eval(dit.parameters())
        }
    }

    public func apply(thinker: Qwen2_5OmniThinker) {
        if let b = encoderBits, thinker.config.quantization == nil {
            thinker.quantizeLayers(groupSize: groupSize, bits: b)
            eval(thinker.parameters())
        }
    }
}

/// How a pipeline holds its two large models (AB-T-0414).
public enum AukResidency: String, Codable, Sendable {
    /// Encoder, DiT and VAE stay loaded for the pipeline's lifetime: the fastest call, the largest footprint.
    case resident
    /// Only the VAE, the fusion head and the tokenizer stay loaded; the encoder and the DiT are read for each call and
    /// freed after their stage. Bit-identical output; each call re-reads both from disk.
    case staged
}

/// One of the pipeline's large models: held for the pipeline's lifetime, or built for one use and dropped after it.
public final class AukStage<Value> {
    private let build: (() throws -> Value)?
    /// The held model, or nil for a staged one (it exists only inside `use`).
    public let held: Value?
    /// Seconds the last `use` spent building the model (0 when held).
    public private(set) var lastBuildSeconds: Double = 0

    public init(resident value: Value) { held = value; build = nil }
    public init(staged build: @escaping () throws -> Value) { held = nil; self.build = build }

    public func use<R>(_ body: (Value) throws -> R) throws -> R {
        if let held { return try body(held) }
        let result = try buildAndUse(body)
        // the staged model's last reference went with buildAndUse's frame; hand its buffers back before the next stage
        Memory.clearCache()
        return result
    }

    private func buildAndUse<R>(_ body: (Value) throws -> R) throws -> R {
        let t0 = Date()
        let value = try build!()
        lastBuildSeconds = Date().timeIntervalSince(t0)
        return try body(value)
    }
}

public final class AukPipeline {
    public let config: AukModelConfig
    public let vae: BigVGANFlowVAE
    public let fusion: AukFusion
    public let dtype: DType
    public let residency: AukResidency
    let dit: AukStage<CFMEdit>
    let encoderStage: AukStage<AukConditionEncoder>

    /// The resident DiT / encoder; nil in `.staged`, where each exists only during its stage of a call.
    public var model: CFMEdit? { dit.held }
    public var encoder: AukConditionEncoder? { encoderStage.held }
    /// Seconds the last call spent reading the staged models from disk (both 0 when resident).
    public var lastStageLoadSeconds: (encoder: Double, dit: Double) { (encoderStage.lastBuildSeconds, dit.lastBuildSeconds) }

    public init(config: AukModelConfig, model: CFMEdit, vae: BigVGANFlowVAE, encoder: AukConditionEncoder, dtype: DType) {
        self.config = config; self.vae = vae; self.fusion = model.fusion; self.dtype = dtype; self.residency = .resident
        self.dit = AukStage(resident: model); self.encoderStage = AukStage(resident: encoder)
    }

    init(config: AukModelConfig, vae: BigVGANFlowVAE, fusion: AukFusion, dtype: DType,
         dit: AukStage<CFMEdit>, encoder: AukStage<AukConditionEncoder>) {
        self.config = config; self.vae = vae; self.fusion = fusion; self.dtype = dtype; self.residency = .staged
        self.dit = dit; self.encoderStage = encoder
    }

    public static func loadModel(_ dir: URL, dtype: DType, numEncoderLayers: Int = 36) throws -> (AukModelConfig, CFMEdit, BigVGANFlowVAE) {
        let config = try ConfigIO.load(AukModelConfig.self, from: dir.appendingPathComponent("config.json"))
        let model = try loadDiT(dir, config: config, dtype: dtype, numEncoderLayers: numEncoderLayers)
        let vae = try loadVAE(dir, config: config, dtype: dtype)
        return (config, model, vae)
    }

    public static func loadDiT(_ dir: URL, config: AukModelConfig, dtype: DType, numEncoderLayers: Int = 36) throws -> CFMEdit {
        let model = CFMEdit(config.arch, numEncoderLayers: numEncoderLayers)
        let keep: (String) -> Bool = { $0.hasSuffix("layer_weights") || $0.hasSuffix("layer_scale") || $0.hasSuffix("inv_freq") }
        try WeightIO.apply(try WeightIO.load(dir.appendingPathComponent("dit.safetensors"), dtype: dtype, keepFP32: keep),
                           to: model, component: "dit")
        return model
    }

    public static func loadVAE(_ dir: URL, config: AukModelConfig, dtype: DType) throws -> BigVGANFlowVAE {
        let vae = BigVGANFlowVAE(config.vae)
        // the VAE stays fp32 below bf16 DiT tiers only if asked; default follows `dtype`, statistics + taps fp32
        let keepVAE: (String) -> Bool = { $0.hasSuffix("global_mean") || $0.hasSuffix("global_log_std") || $0.hasSuffix("filter") }
        try WeightIO.apply(try WeightIO.load(dir.appendingPathComponent("vae.safetensors"), dtype: dtype, keepFP32: keepVAE),
                           to: vae, component: "vae")
        return vae
    }

    public static func load(modelDir: URL, encoderDir: URL, dtype: DType, encoderDType: DType? = nil,
                            quantization: AukQuantization = .none,
                            residency: AukResidency = .resident) async throws -> AukPipeline {
        let encDType = encoderDType ?? dtype
        guard residency == .staged else {
            let (config, model, vae) = try loadModel(modelDir, dtype: dtype)
            let encoder = try await QwenOmniConditionEncoder(directory: encoderDir, dtype: encDType)
            quantization.apply(dit: model, thinker: encoder.thinker)
            return AukPipeline(config: config, model: model, vae: vae, encoder: encoder, dtype: dtype)
        }
        let config = try ConfigIO.load(AukModelConfig.self, from: modelDir.appendingPathComponent("config.json"))
        let vae = try loadVAE(modelDir, config: config, dtype: dtype)
        let fusion = try AukFusion.load(modelDir.appendingPathComponent("dit.safetensors"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: encoderDir)
        return AukPipeline(
            config: config, vae: vae, fusion: fusion, dtype: dtype,
            dit: AukStage(staged: {
                let model = try loadDiT(modelDir, config: config, dtype: dtype)
                quantization.apply(dit: model)
                return model
            }),
            encoder: AukStage(staged: {
                let encoder = try QwenOmniConditionEncoder(directory: encoderDir, dtype: encDType, tokenizer: tokenizer)
                quantization.apply(thinker: encoder.thinker)
                return encoder
            }))
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
        let (textEmbeds, contextMask): (MLXArray, MLXArray) = try encoderStage.use { encoder in
            if let enc = encoder as? QwenOmniConditionEncoder, let feats = inject?.features {
                let w = softmax(fusion.layerWeights.asType(.float32), axis: 0)
                var acc: MLXArray? = nil
                let n = try enc.forEachHiddenState(prompt, featuresOverride: feats) { i, h in
                    guard i >= 1 else { return }
                    let term = MLXFast.layerNorm(h.asType(.float32), weight: nil, bias: nil, eps: 1e-5) * w[i - 1]
                    acc = acc.map { $0 + term } ?? term
                    eval(acc!)
                }
                return (acc! * fusion.layerScale.asType(.float32), MLXArray.ones([1, n]).asType(.bool))
            }
            return try fusion.encodeText(prompt, encoder: encoder)
        }
        eval(textEmbeds)
        capture?("fused", textEmbeds)

        let sampling = config.is_flash ? config.sampling : (req.sampling ?? config.sampling)
        let y0 = inject?.y0 ?? MLXRandom.normal([1, genLen, config.vae.latent_dim], key: MLXRandom.key(req.seed))
        precondition(y0.dim(1) == genLen, "injected y0 has \(y0.dim(1)) frames, request needs \(genLen)")
        let gen = try dit.use { model in
            model.sample(cond: refLatents, refLen: refLatentLen, textEmbeds: textEmbeds, contextMask: contextMask, y0: y0,
                         sampling: sampling, dtype: dtype, onStep: onStep, capture: capture)
        }
        capture?("gen_latent", gen)

        let wave = vae.inferenceFromLatents(vae.denormalize(gen).asType(dtype))
        eval(wave)
        capture?("audio_out", wave)
        return wave.asType(.float32).reshaped(-1).asArray(Float.self)
    }
}
