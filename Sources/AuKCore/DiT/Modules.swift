// Modules.swift — auk/model/modules.py, 1:1 (class and key names kept). Channels-last throughout: upstream permutes
// (B, N, D) → (B, D, N) around its Conv1d; MLX convolutions are NLC, so those permutes disappear and nothing else moves.
//
// Two upstream numerics a port has to copy rather than "fix":
//   • torch.nn.RMSNorm(eps=None) uses torch.finfo(x.dtype).eps — 1.19e-7 for the fp32 oracle (CPU/MPS inference), but
//     2^-7 under CUDA bf16 autocast. We match the fp32 reference (`torchRMSNormEps`).
//   • RoPE is x_transformers' RotaryEmbedding: frequencies duplicated per PAIR ("(d r)") and an interleaved
//     rotate_half — the "traditional" layout — with positions restarting at 0 per stream in the MMDiT blocks.

import Foundation
import MLX
import MLXFast
import MLXNN

public let torchRMSNormEps: Float = 1.1920929e-07

/// torch.nn.RMSNorm(dim, elementwise_affine=True) — key `weight`.
public final class TorchRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    public init(_ dim: Int, eps: Float = torchRMSNormEps) {
        self._weight.wrappedValue = MLXArray.ones([dim])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight.asType(x.dtype), eps: eps)
    }
}

/// nn.LayerNorm(dim, elementwise_affine=False, eps).
@inline(__always)
func plainLayerNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
}

// MARK: - sinusoidal position embedding

public struct SinusPositionEmbedding {
    let dim: Int

    /// x (B,) → (B, dim): [sin | cos] of scale · x · exp(−i · ln(10000) / (half − 1)).
    public func callAsFunction(_ x: MLXArray, scale: Float = 1000) -> MLXArray {
        let half = dim / 2
        let emb = log(10000.0) / Double(half - 1)
        let freqs = exp(MLXArray(0 ..< half).asType(.float32) * Float(-emb))
        let e = (Float(scale) * x.asType(.float32)).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return concatenated([sin(e), cos(e)], axis: -1)
    }
}

// MARK: - convolutional position embedding

public final class ConvPositionEmbedding: Module {
    /// nn.Sequential(Conv1d, Mish, Conv1d, Mish) → keys conv1d.0 / conv1d.2.
    @ModuleInfo(key: "conv1d") var conv1d: [UnaryLayer]

    public init(dim: Int, kernelSize: Int = 31, groups: Int = 16) {
        precondition(kernelSize % 2 != 0)
        self._conv1d.wrappedValue = [
            Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, padding: kernelSize / 2, groups: groups),
            Mish(),
            Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, padding: kernelSize / 2, groups: groups),
            Mish(),
        ]
    }

    /// x (B, N, D), mask (B, N) bool.
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let m = mask.map { $0.expandedDimensions(axis: -1).asType(x.dtype) }
        var x = x
        if let m { x = x * m }
        for block in conv1d {
            x = block(x)
            if let m, block is Conv1d { x = x * m }
        }
        return x
    }
}

// MARK: - AdaLayerNorm

public final class AdaLayerNorm: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    public init(dim: Int) {
        self._linear.wrappedValue = Linear(dim, dim * 6)
    }

    /// Returns (modulated x, gate_msa, shift_mlp, scale_mlp, gate_mlp); emb (B, D).
    public func callAsFunction(_ x: MLXArray, emb: MLXArray)
        -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let e = linear(silu(emb))
        let p = split(e, parts: 6, axis: 1)  // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        let x = plainLayerNorm(x, eps: 1e-6) * (1 + p[1].expandedDimensions(axis: 1)) + p[0].expandedDimensions(axis: 1)
        return (x, p[2], p[3], p[4], p[5])
    }
}

public final class AdaLayerNorm_Final: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    public init(dim: Int) {
        self._linear.wrappedValue = Linear(dim, dim * 2)
    }

    public func callAsFunction(_ x: MLXArray, emb: MLXArray) -> MLXArray {
        let e = linear(silu(emb))
        let p = split(e, parts: 2, axis: 1)  // scale, shift
        return plainLayerNorm(x, eps: 1e-6) * (1 + p[0].expandedDimensions(axis: 1)) + p[1].expandedDimensions(axis: 1)
    }
}

// MARK: - SwiGLU feed-forward

public final class SwiGLUFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    public init(dim: Int, dimOut: Int? = nil, mult: Double = 3.0) {
        let inner = Int(Double(dim) * mult)
        self._linearIn.wrappedValue = Linear(dim, inner * 2, bias: false)
        self._linearOut.wrappedValue = Linear(inner, dimOut ?? dim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = split(linearIn(x), parts: 2, axis: -1)
        return linearOut(silu(h[0]) * h[1])
    }
}

// MARK: - rotary (x_transformers RotaryEmbedding, no xpos)

public final class RotaryEmbedding: Module {
    @ParameterInfo(key: "inv_freq") var invFreq: MLXArray

    public init(dim: Int, base: Double = 10000) {
        let half = dim / 2
        self._invFreq.wrappedValue = MLXArray((0 ..< half).map { Float(1.0 / pow(base, Double(2 * $0) / Double(dim))) })
    }

    /// (1, N, dim) frequencies, each duplicated per pair — `forward_from_seq_len`.
    public func forwardFromSeqLen(_ n: Int) -> MLXArray {
        let t = MLXArray(0 ..< n).asType(.float32)
        let f = t.expandedDimensions(axis: 1) * invFreq.asType(.float32).expandedDimensions(axis: 0)  // (N, half)
        let pair = stacked([f, f], axis: -1).reshaped(n, -1)                                           // (N, dim)
        return pair.expandedDimensions(axis: 0)
    }
}

/// x_transformers.apply_rotary_pos_emb with scale 1: t (B, H, N, D), freqs (1, N, D); interleaved rotate_half.
func applyRotaryPosEmb(_ t: MLXArray, _ freqs: MLXArray) -> MLXArray {
    let n = t.dim(-2)
    let f = freqs[0..., (freqs.dim(1) - n)...].expandedDimensions(axis: 1).asType(.float32)  // (1, 1, N, D)
    let tf = t.asType(.float32)
    let shape = tf.shape
    let pairs = tf.reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2])
    let x1 = pairs[.ellipsis, 0], x2 = pairs[.ellipsis, 1]
    let rot = stacked([-x2, x1], axis: -1).reshaped(shape)
    return (tf * cos(f) + rot * sin(f)).asType(t.dtype)
}

// MARK: - attention

/// (B, N) bool key mask → additive (B, 1, 1, N) mask, or nil when every key is valid (the batch-1 inference case).
func additiveKeyMask(_ mask: MLXArray?, dtype: DType) -> MLXArray? {
    guard let mask else { return nil }
    if mask.all().item(Bool.self) { return nil }
    let neg = MLXArray(-Float.greatestFiniteMagnitude).asType(dtype)
    return MLX.where(mask, MLXArray(0, dtype: dtype), neg).expandedDimensions(axes: [1, 2])
}

public final class Attention: Module {
    let heads: Int
    let innerDim: Int
    let isJoint: Bool
    let attnMaskEnabled: Bool

    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "q_norm") var qNorm: TorchRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: TorchRMSNorm
    @ModuleInfo(key: "to_qkv_c") var toQKVc: Linear?
    @ModuleInfo(key: "c_q_norm") var cQNorm: TorchRMSNorm?
    @ModuleInfo(key: "c_k_norm") var cKNorm: TorchRMSNorm?
    /// nn.ModuleList([Linear, Dropout]) → key to_out.0 (dropout is identity at inference).
    @ModuleInfo(key: "to_out") var toOut: [UnaryLayer]
    @ModuleInfo(key: "to_out_c") var toOutC: Linear?

    public init(dim: Int, heads: Int, dimHead: Int, contextDim: Int? = nil, attnMaskEnabled: Bool) {
        self.heads = heads
        self.innerDim = dimHead * heads
        self.isJoint = contextDim != nil
        self.attnMaskEnabled = attnMaskEnabled
        self._toQKV.wrappedValue = Linear(dim, 3 * innerDim)
        self._qNorm.wrappedValue = TorchRMSNorm(dimHead)
        self._kNorm.wrappedValue = TorchRMSNorm(dimHead)
        if let c = contextDim {
            self._toQKVc.wrappedValue = Linear(c, 3 * innerDim)
            self._cQNorm.wrappedValue = TorchRMSNorm(dimHead)
            self._cKNorm.wrappedValue = TorchRMSNorm(dimHead)
            self._toOutC.wrappedValue = Linear(innerDim, c)
        }
        self._toOut.wrappedValue = [Linear(innerDim, dim), Dropout(p: 0)]
    }

    private func heads(_ t: MLXArray, _ b: Int) -> MLXArray {
        t.reshaped(b, -1, heads, innerDim / heads).transposed(0, 2, 1, 3)  // (B, H, N, D)
    }

    private func merge(_ t: MLXArray, _ b: Int) -> MLXArray {
        t.transposed(0, 2, 1, 3).reshaped(b, -1, innerDim)
    }

    /// AttnProcessor — self attention over x.
    public func callAsFunction(_ x: MLXArray, mask: MLXArray?, rope: MLXArray?) -> MLXArray {
        let b = x.dim(0)
        let qkv = split(toQKV(x), parts: 3, axis: -1)
        var q = qNorm(heads(qkv[0], b)), k = kNorm(heads(qkv[1], b))
        let v = heads(qkv[2], b)
        if let rope { q = applyRotaryPosEmb(q, rope); k = applyRotaryPosEmb(k, rope) }
        let m = attnMaskEnabled ? additiveKeyMask(mask, dtype: q.dtype) : nil
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(innerDim / heads).squareRoot(), mask: m)
        var out = toOut[0](merge(o, b))
        if let mask { out = out * mask.expandedDimensions(axis: -1).asType(out.dtype) }
        return out
    }

    /// JointAttnProcessor — joint attention, sequence order [x ; c].
    public func joint(_ x: MLXArray, c: MLXArray, mask: MLXArray?, rope: MLXArray?, cRope: MLXArray?, cMask: MLXArray?)
        -> (MLXArray, MLXArray)
    {
        let b = c.dim(0)
        let qkv = split(toQKV(x), parts: 3, axis: -1)
        let cqkv = split(toQKVc!(c), parts: 3, axis: -1)
        var q = qNorm(heads(qkv[0], b)), k = kNorm(heads(qkv[1], b))
        let v = heads(qkv[2], b)
        var cq = cQNorm!(heads(cqkv[0], b)), ck = cKNorm!(heads(cqkv[1], b))
        let cv = heads(cqkv[2], b)
        if let rope { q = applyRotaryPosEmb(q, rope); k = applyRotaryPosEmb(k, rope) }
        if let cRope { cq = applyRotaryPosEmb(cq, cRope); ck = applyRotaryPosEmb(ck, cRope) }
        let qj = concatenated([q, cq], axis: 2), kj = concatenated([k, ck], axis: 2), vj = concatenated([v, cv], axis: 2)

        var m: MLXArray? = nil
        if attnMaskEnabled, let mask {
            let joint = cMask.map { concatenated([mask, $0], axis: 1) }
                ?? padded(mask, widths: [IntOrPair((0, 0)), IntOrPair((0, c.dim(1)))], value: MLXArray(true))
            m = additiveKeyMask(joint, dtype: q.dtype)
        }
        let o = merge(MLXFast.scaledDotProductAttention(
            queries: qj, keys: kj, values: vj, scale: 1 / Float(innerDim / heads).squareRoot(), mask: m), b)
        let nx = x.dim(1)
        var xo = toOut[0](o[0..., ..<nx])
        var co = toOutC!(o[0..., nx...])
        if let mask { xo = xo * mask.expandedDimensions(axis: -1).asType(xo.dtype) }
        if let cMask { co = co * cMask.expandedDimensions(axis: -1).asType(co.dtype) }
        return (xo, co)
    }
}

// MARK: - blocks

public final class DiTBlock: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: AdaLayerNorm
    @ModuleInfo(key: "attn") var attn: Attention
    @ModuleInfo(key: "ff") var ff: SwiGLUFeedForward

    public init(dim: Int, heads: Int, dimHead: Int, ffMult: Double, attnMaskEnabled: Bool) {
        self._attnNorm.wrappedValue = AdaLayerNorm(dim: dim)
        self._attn.wrappedValue = Attention(dim: dim, heads: heads, dimHead: dimHead, attnMaskEnabled: attnMaskEnabled)
        self._ff.wrappedValue = SwiGLUFeedForward(dim: dim, mult: ffMult)
    }

    public func callAsFunction(_ x: MLXArray, t: MLXArray, mask: MLXArray?, rope: MLXArray?) -> MLXArray {
        let (norm, gateMsa, shiftMlp, scaleMlp, gateMlp) = attnNorm(x, emb: t)
        let attnOutput = attn(norm, mask: mask, rope: rope)
        var x = x + gateMsa.expandedDimensions(axis: 1) * attnOutput
        let n2 = plainLayerNorm(x, eps: 1e-6) * (1 + scaleMlp.expandedDimensions(axis: 1)) + shiftMlp.expandedDimensions(axis: 1)
        x = x + gateMlp.expandedDimensions(axis: 1) * ff(n2)
        return x
    }
}

public final class MMDiTBlock: Module {
    @ModuleInfo(key: "attn_norm_c") var attnNormC: AdaLayerNorm
    @ModuleInfo(key: "attn_norm_x") var attnNormX: AdaLayerNorm
    @ModuleInfo(key: "attn") var attn: Attention
    @ModuleInfo(key: "ff_c") var ffC: SwiGLUFeedForward
    @ModuleInfo(key: "ff_x") var ffX: SwiGLUFeedForward

    public init(dim: Int, heads: Int, dimHead: Int, ffMult: Double, attnMaskEnabled: Bool) {
        self._attnNormC.wrappedValue = AdaLayerNorm(dim: dim)
        self._attnNormX.wrappedValue = AdaLayerNorm(dim: dim)
        self._attn.wrappedValue = Attention(
            dim: dim, heads: heads, dimHead: dimHead, contextDim: dim, attnMaskEnabled: attnMaskEnabled)
        self._ffC.wrappedValue = SwiGLUFeedForward(dim: dim, mult: ffMult)
        self._ffX.wrappedValue = SwiGLUFeedForward(dim: dim, mult: ffMult)
    }

    /// Returns (c, x), as upstream.
    public func callAsFunction(
        _ x: MLXArray, c: MLXArray, t: MLXArray, mask: MLXArray?, rope: MLXArray?, cRope: MLXArray?, cMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let (normC, cGateMsa, cShiftMlp, cScaleMlp, cGateMlp) = attnNormC(c, emb: t)
        let (normX, xGateMsa, xShiftMlp, xScaleMlp, xGateMlp) = attnNormX(x, emb: t)
        let (xAttn, cAttn) = attn.joint(normX, c: normC, mask: mask, rope: rope, cRope: cRope, cMask: cMask)

        var c = c + cGateMsa.expandedDimensions(axis: 1) * cAttn
        let nc = plainLayerNorm(c, eps: 1e-6) * (1 + cScaleMlp.expandedDimensions(axis: 1)) + cShiftMlp.expandedDimensions(axis: 1)
        c = c + cGateMlp.expandedDimensions(axis: 1) * ffC(nc)

        var x = x + xGateMsa.expandedDimensions(axis: 1) * xAttn
        let nx = plainLayerNorm(x, eps: 1e-6) * (1 + xScaleMlp.expandedDimensions(axis: 1)) + xShiftMlp.expandedDimensions(axis: 1)
        x = x + xGateMlp.expandedDimensions(axis: 1) * ffX(nx)
        return (c, x)
    }
}

// MARK: - time step embedding

public final class TimestepEmbedding: Module {
    let timeEmbed: SinusPositionEmbedding
    /// nn.Sequential(Linear, SiLU, Linear) → keys time_mlp.0 / time_mlp.2.
    @ModuleInfo(key: "time_mlp") var timeMlp: [UnaryLayer]

    public init(dim: Int, freqEmbedDim: Int = 256) {
        self.timeEmbed = SinusPositionEmbedding(dim: freqEmbedDim)
        self._timeMlp.wrappedValue = [Linear(freqEmbedDim, dim), SiLU(), Linear(dim, dim)]
    }

    public func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        var h = timeEmbed(timestep).asType(timestep.dtype)
        for l in timeMlp { h = l(h) }
        return h
    }
}
