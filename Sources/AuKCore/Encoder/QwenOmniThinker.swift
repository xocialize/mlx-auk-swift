// QwenOmniThinker.swift — the parts of transformers' Qwen2_5OmniThinkerForConditionalGeneration (4.57.1) that AuK
// runs: the Qwen2 text model for ALL hidden states, and the Whisper-shaped audio tower whose embeddings are scattered
// into the <|AUDIO|> positions. No vision tower, no LM head, no generation. Keys equal the bundle's `thinker.*` names
// (the converter kept `model.*` and `audio_tower.*`).
//
// Facts this file depends on (PORTING-SPEC §2, read from the HF code and confirmed in the V0 goldens):
//   • with audio + text only, get_rope_index returns cumsum(attention_mask) − 1 for all three M-RoPE sections, so the
//     LLM is a plain Qwen2 with 1-D RoPE (θ 1e6, half-split rotate_half → MLXFast.RoPE traditional: false);
//   • hidden_states has 37 entries — the embeddings, layers 1…35, then layer 36 AFTER the final norm;
//   • the audio tower runs 200-frame mel chunks independently (conv stem, sinusoid restarted per chunk, attention
//     block-diagonal per chunk), then AvgPool(2) + ln_post + proj over the whole clip.

import Foundation
import MLX
import MLXFast
import MLXNN

public struct QwenOmniTextConfig: Codable, Sendable {
    public var hidden_size: Int
    public var intermediate_size: Int
    public var num_attention_heads: Int
    public var num_hidden_layers: Int
    public var num_key_value_heads: Int
    public var rms_norm_eps: Double
    public var rope_theta: Double
    public var vocab_size: Int
}

public struct QwenOmniAudioConfig: Codable, Sendable {
    public var d_model: Int
    public var encoder_attention_heads: Int
    public var encoder_ffn_dim: Int
    public var encoder_layers: Int
    public var num_mel_bins: Int
    public var max_source_positions: Int
    public var n_window: Int
    public var output_dim: Int
}

public struct QuantizationConfig: Codable, Sendable, Equatable {
    public var group_size: Int
    public var bits: Int
    public init(group_size: Int, bits: Int) { self.group_size = group_size; self.bits = bits }
}

public struct QwenOmniThinkerConfig: Codable, Sendable {
    public var text_config: QwenOmniTextConfig
    public var audio_config: QwenOmniAudioConfig
    public var audio_token_index: Int
    public var audio_start_token_id: Int
    public var audio_end_token_id: Int
    /// Present on a pre-quantised bundle (mlx-community `-8bit`): the layer Linears are stored as weight/scales/biases.
    public var quantization: QuantizationConfig?
}

// MARK: - Qwen2 text model

final class Qwen2Attention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, theta: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ c: QwenOmniTextConfig) {
        heads = c.num_attention_heads; kvHeads = c.num_key_value_heads
        headDim = c.hidden_size / c.num_attention_heads; theta = Float(c.rope_theta)
        _qProj.wrappedValue = Linear(c.hidden_size, heads * headDim, bias: true)
        _kProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: true)
        _vProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: true)
        _oProj.wrappedValue = Linear(heads * headDim, c.hidden_size, bias: false)
    }

    /// x (B, L, D); positions are 0…L−1 (single unpadded sequence); causal.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: 0)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: 0)
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: .causal)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class Qwen2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: QwenOmniTextConfig) {
        _gateProj.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _upProj.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _downProj.wrappedValue = Linear(c.intermediate_size, c.hidden_size, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

final class Qwen2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen2Attention
    @ModuleInfo(key: "mlp") var mlp: Qwen2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ c: QwenOmniTextConfig) {
        _selfAttn.wrappedValue = Qwen2Attention(c)
        _mlp.wrappedValue = Qwen2MLP(c)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x))
        return h + mlp(postAttentionLayernorm(h))
    }
}

public final class Qwen2_5OmniThinkerTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: QwenOmniTextConfig) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocab_size, dimensions: c.hidden_size)
        _layers.wrappedValue = (0 ..< c.num_hidden_layers).map { _ in Qwen2DecoderLayer(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
    }

    /// Runs the stack on `inputsEmbeds` (B, L, D) and hands every hidden state to `visit(index, h)` in HF order:
    /// 0 = embeddings, 1…N−1 = layer outputs, N = final layer output after `norm`. Nothing is retained.
    public func forEachHiddenState(_ inputsEmbeds: MLXArray, visit: (Int, MLXArray) -> Void) {
        var h = inputsEmbeds
        visit(0, h)
        for (i, layer) in layers.enumerated() {
            h = layer(h)
            visit(i + 1, i == layers.count - 1 ? norm(h) : h)
        }
    }
}

// MARK: - audio tower

final class Qwen2_5OmniAudioAttention: Module {
    let heads: Int, headDim: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: QwenOmniAudioConfig) {
        heads = c.encoder_attention_heads; headDim = c.d_model / c.encoder_attention_heads
        _qProj.wrappedValue = Linear(c.d_model, c.d_model, bias: true)
        _kProj.wrappedValue = Linear(c.d_model, c.d_model, bias: false)
        _vProj.wrappedValue = Linear(c.d_model, c.d_model, bias: true)
        _outProj.wrappedValue = Linear(c.d_model, c.d_model, bias: true)
    }

    /// x (nChunks, L, D): each batch row is one chunk, so batched attention IS the block-diagonal cu_seqlens attention.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return outProj(o.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class Qwen2_5OmniAudioEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen2_5OmniAudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ c: QwenOmniAudioConfig) {
        _selfAttn.wrappedValue = Qwen2_5OmniAudioAttention(c)
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: c.d_model, eps: 1e-5)
        _fc1.wrappedValue = Linear(c.d_model, c.encoder_ffn_dim)
        _fc2.wrappedValue = Linear(c.encoder_ffn_dim, c.d_model)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: c.d_model, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(selfAttnLayerNorm(x))
        return h + fc2(gelu(fc1(finalLayerNorm(h))))
    }
}

final class AudioConv1d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray   // (O, K, I)
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int, padding: Int

    init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int, padding: Int) {
        _weight.wrappedValue = MLXArray.zeros([outC, kernel, inC])
        _bias.wrappedValue = MLXArray.zeros([outC])
        self.stride = stride; self.padding = padding
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv1d(x, weight, stride: stride, padding: padding) + bias }
}

public final class Qwen2_5OmniAudioEncoder: Module {
    let c: QwenOmniAudioConfig
    let _positionalEmbedding: MLXArray   // (max_source_positions, d_model), derived — not on disk

    @ModuleInfo(key: "conv1") var conv1: AudioConv1d
    @ModuleInfo(key: "conv2") var conv2: AudioConv1d
    @ModuleInfo(key: "layers") var layers: [Qwen2_5OmniAudioEncoderLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ c: QwenOmniAudioConfig) {
        self.c = c
        // SinusoidsPositionEmbedding(length, channels, max_timescale=10000), float32 as torch builds it
        let half = c.d_model / 2
        let inc = log(10000.0) / Double(half - 1)
        let inv = exp(MLXArray(0 ..< half).asType(.float32) * Float(-inc))
        let scaled = MLXArray(0 ..< c.max_source_positions).asType(.float32).expandedDimensions(axis: 1) * inv.expandedDimensions(axis: 0)
        _positionalEmbedding = concatenated([sin(scaled), cos(scaled)], axis: 1)
        _conv1.wrappedValue = AudioConv1d(c.num_mel_bins, c.d_model, kernel: 3, stride: 1, padding: 1)
        _conv2.wrappedValue = AudioConv1d(c.d_model, c.d_model, kernel: 3, stride: 2, padding: 1)
        _layers.wrappedValue = (0 ..< c.encoder_layers).map { _ in Qwen2_5OmniAudioEncoderLayer(c) }
        _lnPost.wrappedValue = LayerNorm(dimensions: c.d_model, eps: 1e-5)
        _proj.wrappedValue = Linear(c.d_model, c.output_dim)
    }

    /// Token count the processor reserves for `melFrames` valid frames.
    public static func outputLength(melFrames l: Int) -> Int { (((l - 1) / 2 + 1) - 2) / 2 + 1 }

    /// features (T, n_mels) of ONE clip's valid frames → (T_tokens, output_dim).
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        let total = features.dim(0), w = c.n_window * 2
        var starts = Array(stride(from: 0, to: total, by: w))
        if starts.isEmpty { starts = [0] }
        // full chunks run as one batch; the shorter tail as another (identical to the zero-padded, masked HF batch)
        let full = starts.filter { $0 + w <= total }
        let tailStart = starts.last! + w > total ? starts.last! : nil

        func stem(_ x: MLXArray) -> MLXArray {           // x (n, l, mel)
            var h = gelu(conv1(x.asType(conv1.weight.dtype)))
            h = gelu(conv2(h))
            return h + _positionalEmbedding[..<h.dim(1)].asType(h.dtype)
        }
        func run(_ x: MLXArray) -> MLXArray {
            var h = x
            for l in layers { h = l(h) }
            return h
        }

        var pieces = [MLXArray]()
        if !full.isEmpty {
            let batch = stacked(full.map { features[$0 ..< ($0 + w)] }, axis: 0)
            let out = run(stem(batch))                     // (nFull, 100, D)
            pieces.append(out.reshaped(-1, out.dim(2)))
        }
        if let s = tailStart {
            let out = run(stem(features[s...].expandedDimensions(axis: 0)))
            pieces.append(out[0])
        }
        let h = concatenated(pieces, axis: 0)               // (S, D), time order
        let pooledLen = h.dim(0) / 2
        let pooled = h[..<(pooledLen * 2)].reshaped(pooledLen, 2, -1).mean(axis: 1)   // AvgPool1d(2, 2), floor
        return proj(lnPost(pooled))
    }
}

// MARK: - thinker

public final class Qwen2_5OmniThinker: Module {
    public let config: QwenOmniThinkerConfig
    @ModuleInfo(key: "model") public var model: Qwen2_5OmniThinkerTextModel
    @ModuleInfo(key: "audio_tower") public var audioTower: Qwen2_5OmniAudioEncoder

    public init(_ c: QwenOmniThinkerConfig) {
        config = c
        _model.wrappedValue = Qwen2_5OmniThinkerTextModel(c.text_config)
        _audioTower.wrappedValue = Qwen2_5OmniAudioEncoder(c.audio_config)
    }

    /// The Linears a quantised tier replaces: every Linear inside the LLM and audio-tower layers (PORTING-SPEC §6).
    public static func isQuantizedLayer(_ path: String, _ m: Module) -> Bool {
        m is Linear && (path.hasPrefix("model.layers.") || path.hasPrefix("audio_tower.layers."))
    }

    /// Swap the layer Linears for QuantizedLinear (structure only — weights come after).
    public func quantizeLayers(groupSize: Int, bits: Int) {
        quantize(model: self, groupSize: groupSize, bits: bits) { Self.isQuantizedLayer($0, $1) }
    }

    /// Token embeddings with the audio tower's output scattered into the <|AUDIO|> positions (B = 1).
    public func inputsEmbeds(inputIds: [Int32], audioEmbeds: MLXArray?) -> MLXArray {
        let ids = MLXArray(inputIds).expandedDimensions(axis: 0)
        var e = model.embedTokens(ids)                       // (1, L, D)
        guard let audioEmbeds else { return e }
        let positions = inputIds.enumerated().filter { Int($0.element) == config.audio_token_index }.map(\.offset)
        precondition(positions.count == audioEmbeds.dim(0),
                     "audio placeholder count \(positions.count) != audio tokens \(audioEmbeds.dim(0))")
        guard let first = positions.first else { return e }
        // the processor emits the placeholders contiguously
        precondition(positions.last! - first + 1 == positions.count)
        e = concatenated([e[0..., ..<first], audioEmbeds.asType(e.dtype).expandedDimensions(axis: 0),
                          e[0..., (positions.last! + 1)...]], axis: 1)
        return e
    }
}
