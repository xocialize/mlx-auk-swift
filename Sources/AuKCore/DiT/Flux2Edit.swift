// Flux2Edit.swift — auk/model/flux2_edit.py, 1:1: AudioPromptEmbedding + the two-phase backbone (10 MMDiT blocks with
// joint text–audio attention, then 20 single-stream blocks over [text ; ref ; target]). Keys equal the checkpoint's
// `transformer.*` names.

import Foundation
import MLX
import MLXNN

public struct Flux2EditConfig: Codable, Sendable {
    public var dim: Int
    public var heads: Int
    public var dim_head: Int
    public var ff_mult: Double
    public var text_hidden_dim: Int
    public var attn_mask_enabled: Bool
    public var num_layers: Int
    public var num_single_layers: Int
    public var latent_dim: Int
}

public final class AudioPromptEmbedding: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "conv_pos_embed") var convPosEmbed: ConvPositionEmbedding

    public init(inDim: Int, outDim: Int) {
        self._linear.wrappedValue = Linear(inDim, outDim)
        self._convPosEmbed.wrappedValue = ConvPositionEmbedding(dim: outDim)
    }

    func embed(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let x = linear(x)
        return convPosEmbed(x, mask: mask) + x
    }
}

/// Holds the projected text across sampler steps (`text_cond` / `text_uncond`) outside the module tree.
final class TextCache {
    var cond: MLXArray?
    var uncond: MLXArray?
}

public final class Flux2Edit: Module {
    public let config: Flux2EditConfig

    @ModuleInfo(key: "time_embed") var timeEmbed: TimestepEmbedding
    @ModuleInfo(key: "txt_norm") var txtNorm: TorchRMSNorm
    @ModuleInfo(key: "txt_proj") var txtProj: Linear
    @ModuleInfo(key: "audio_embed") var audioEmbed: AudioPromptEmbedding
    @ModuleInfo(key: "rotary_embed") var rotaryEmbed: RotaryEmbedding
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [MMDiTBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [DiTBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNorm_Final
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let cache = TextCache()

    public init(_ c: Flux2EditConfig) {
        self.config = c
        self._timeEmbed.wrappedValue = TimestepEmbedding(dim: c.dim)
        self._txtNorm.wrappedValue = TorchRMSNorm(c.dim)
        self._txtProj.wrappedValue = Linear(c.text_hidden_dim, c.dim)
        self._audioEmbed.wrappedValue = AudioPromptEmbedding(inDim: c.latent_dim, outDim: c.dim)
        self._rotaryEmbed.wrappedValue = RotaryEmbedding(dim: c.dim_head)
        self._transformerBlocks.wrappedValue = (0 ..< c.num_layers).map { _ in
            MMDiTBlock(dim: c.dim, heads: c.heads, dimHead: c.dim_head, ffMult: c.ff_mult, attnMaskEnabled: c.attn_mask_enabled)
        }
        self._singleTransformerBlocks.wrappedValue = (0 ..< c.num_single_layers).map { _ in
            DiTBlock(dim: c.dim, heads: c.heads, dimHead: c.dim_head, ffMult: c.ff_mult, attnMaskEnabled: c.attn_mask_enabled)
        }
        self._normOut.wrappedValue = AdaLayerNorm_Final(dim: c.dim)
        self._projOut.wrappedValue = Linear(c.dim, c.latent_dim)
    }

    /// Linear → RMSNorm; zeros when the text is dropped.
    func projectText(_ text: MLXArray, dropText: Bool) -> MLXArray {
        let c = txtNorm(txtProj(text))
        return dropText ? MLXArray.zeros(like: c) : c
    }

    public func clearCache() {
        cache.cond = nil
        cache.uncond = nil
    }

    /// Returns (audio stream [ref ; x], its mask, prompt length).
    func embedAudio(_ x: MLXArray, ref: MLXArray?, dropAudioCond: Bool, mask: MLXArray?, refMask: MLXArray?)
        -> (MLXArray, MLXArray?, Int)
    {
        let xEmb = audioEmbed.embed(x, mask: mask)
        guard var ref, ref.dim(1) > 0 else { return (xEmb, mask, 0) }
        if dropAudioCond { ref = MLXArray.zeros(like: ref) }
        let refEmb = audioEmbed.embed(ref, mask: refMask)
        let audio = concatenated([refEmb, xEmb], axis: 1)
        var audioMask: MLXArray? = nil
        if mask != nil || refMask != nil {
            let (b, n, p) = (xEmb.dim(0), xEmb.dim(1), refEmb.dim(1))
            audioMask = concatenated([refMask ?? MLXArray.ones([b, p]).asType(.bool), mask ?? MLXArray.ones([b, n]).asType(.bool)], axis: 1)
        }
        return (audio, audioMask, refEmb.dim(1))
    }

    /// One velocity evaluation. With `cfgInfer`, the batch is doubled to [cond ; uncond] and the caller splits it.
    public func callAsFunction(
        x: MLXArray, text: MLXArray, time: MLXArray, mask: MLXArray? = nil, cMask: MLXArray?,
        ref: MLXArray?, refMask: MLXArray?, dropAudioCond: Bool = false, dropText: Bool = false,
        cfgInfer: Bool = false, cache useCache: Bool = false,
        capture: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let time = time.ndim == 0 ? repeated(time.expandedDimensions(axis: 0), count: batch, axis: 0) : time
        var t = timeEmbed(time)
        var cMask = cMask ?? (abs(text).sum(axis: -1) .> 0)

        var xs: MLXArray, c: MLXArray, audioMask: MLXArray?, promptLen: Int
        if cfgInfer {
            let cCond: MLXArray
            if useCache, let cached = cache.cond { cCond = cached } else {
                cCond = projectText(text, dropText: false)
                if useCache { cache.cond = cCond }
            }
            let (xCond, aMaskCond, pl) = embedAudio(x, ref: ref, dropAudioCond: false, mask: mask, refMask: refMask)
            let cUncond: MLXArray
            if useCache, let cached = cache.uncond { cUncond = cached } else {
                cUncond = projectText(text, dropText: true)
                if useCache { cache.uncond = cUncond }
            }
            let (xUncond, aMaskUncond, _) = embedAudio(x, ref: ref, dropAudioCond: true, mask: mask, refMask: refMask)
            xs = concatenated([xCond, xUncond], axis: 0)
            c = concatenated([cCond, cUncond], axis: 0)
            t = concatenated([t, t], axis: 0)
            if let a = aMaskCond, let b = aMaskUncond { audioMask = concatenated([a, b], axis: 0) } else { audioMask = nil }
            cMask = concatenated([cMask, cMask], axis: 0)
            promptLen = pl
        } else {
            c = projectText(text, dropText: dropText)
            (xs, audioMask, promptLen) = embedAudio(x, ref: ref, dropAudioCond: dropAudioCond, mask: mask, refMask: refMask)
        }

        let seqLen = xs.dim(1), textLen = c.dim(1)
        let ropeAudio = rotaryEmbed.forwardFromSeqLen(seqLen)
        let ropeText = rotaryEmbed.forwardFromSeqLen(textLen)

        for (i, block) in transformerBlocks.enumerated() {
            (c, xs) = block(xs, c: c, t: t, mask: audioMask, rope: ropeAudio, cRope: ropeText, cMask: cMask)
            capture?("block\(i)_c", c)
            capture?("block\(i)_x", xs)
        }

        var h = concatenated([c, xs], axis: 1)
        let rope = rotaryEmbed.forwardFromSeqLen(textLen + seqLen)
        let singleMask = audioMask.map { concatenated([cMask, $0], axis: 1) }
        for (i, block) in singleTransformerBlocks.enumerated() {
            h = block(h, t: t, mask: singleMask, rope: rope)
            capture?("single\(i)", h)
        }

        h = h[0..., (textLen + promptLen)...]
        h = normOut(h, emb: t)
        return projOut(h)
    }
}
