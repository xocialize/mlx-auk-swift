// BigVGANFlowVAE.swift — auk/model/vae/bigvgan_flow_vae.py (+ modules/commons/layers.py, bigvgan/activations.py,
// bigvgan/alias_free_torch/*), inference subset: `encoding_and_normalization`, `denormalize`, `inference_from_latents`.
// The coupling `flow` is training-only and absent from the bundle. Weight-norms arrive folded, conv weights in MLX
// (O, K, I) layout; tensors are channels-last (B, T, C) where upstream is (B, C, T).
//
// Upstream details kept on purpose:
//   • encoder ResStack uses nn.LeakyReLU() — slope 0.01 — while the trunk uses 0.2;
//   • the decoder is causal (left-padded convs; ConvTranspose k = 2·stride trimmed by `stride` at the end) except
//     conv_pre (symmetric pad 3);
//   • in Activation1d the UP-sampler is non-causal (replicate pad 5, trim 15 / 15) even though act_causal=True — only
//     the DOWN-sampler is built causal (replicate left pad 11);
//   • the anti-alias filters are the checkpoint's own kaiser-sinc taps (`*.filter`), not recomputed.

import Foundation
import MLX
import MLXNN

public struct BigVGANFlowVAEConfig: Codable, Sendable {
    public var latent_dim: Int
    public var downsample_rate: Int
    public var target_sample_rate: Int
    public var upsample_rates: [Int]
    public var upsample_kernel_sizes: [Int]
    public var upsample_initial_channel: Int
    public var resblock_kernel_sizes: [Int]
    public var resblock_dilation_sizes: [[Int]]
    public var downsample_rates: [Int]
    public var downsample_channels: [Int]
    public var snake_logscale: Bool
    public var use_vae: Bool
    public var causal: Bool
    public var act_causal: Bool
}

// MARK: - conv primitives

/// commons.layers.Conv1d (and the plain nn.Conv1d inside Conv1d_S / ResStack): weight (O, K, I), optional bias,
/// causal = left pad dilation·(k−1), else symmetric `padding`.
public final class VConv1d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    let stride: Int, dilation: Int, padding: Int, causal: Bool, kernel: Int

    public init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int = 1, dilation: Int = 1, padding: Int? = nil,
                causal: Bool = false, bias: Bool = true) {
        self._weight.wrappedValue = MLXArray.zeros([outC, kernel, inC])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outC]) : nil
        self.stride = stride; self.dilation = dilation; self.kernel = kernel; self.causal = causal
        self.padding = padding ?? (causal ? 0 : (kernel * dilation - dilation) / 2)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if causal {
            x = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((dilation * (kernel - 1), 0)), IntOrPair((0, 0))])
        }
        var y = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
        if let bias { y = y + bias }
        return y
    }
}

/// commons.layers.ConvTranspose1d, causal: padding 0, output trimmed by `stride` at the end. weight (O, K, I).
public final class VConvTranspose1d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int, causal: Bool, kernel: Int

    public init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int, causal: Bool) {
        self._weight.wrappedValue = MLXArray.zeros([outC, kernel, inC])
        self._bias.wrappedValue = MLXArray.zeros([outC])
        self.stride = stride; self.causal = causal; self.kernel = kernel
        precondition(!causal || kernel == 2 * stride, "causal ConvTranspose1d needs kernel == 2·stride")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pad = causal ? 0 : (kernel - stride) / 2
        var y = convTransposed1d(x, weight, stride: stride, padding: pad) + bias
        if causal { y = y[0..., ..<(y.dim(1) - stride), 0...] }
        return y
    }
}

/// Depthwise conv of every channel with one shared 1-D filter: x (B, T, C), taps (K) → (B, T', C).
func depthwise(_ x: MLXArray, taps: MLXArray, stride: Int) -> MLXArray {
    let c = x.dim(2), k = taps.dim(0)
    let w = broadcast(taps.reshaped(1, k, 1), to: [c, k, 1]).asType(x.dtype)
    return conv1d(x, w, stride: stride, padding: 0, groups: c)
}

// MARK: - alias-free activation

public final class SnakeBeta: Module, UnaryLayer {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray
    let logscale: Bool

    public init(_ channels: Int, logscale: Bool) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = MLXArray.zeros([channels])
        self.logscale = logscale
    }

    /// x + 1/(β + 1e-9) · sin²(α·x), per channel (last axis).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = logscale ? exp(alpha) : alpha
        let b = logscale ? exp(beta) : beta
        let s = sin(x * a)
        return x + (1.0 / (b + 1e-9)) * (s * s)
    }
}

public final class UpSample1d: Module {
    @ParameterInfo(key: "filter") var filter: MLXArray   // (1, 1, K) as saved
    let ratio: Int, kernel: Int

    public init(ratio: Int = 2, kernel: Int = 12) {
        self._filter.wrappedValue = MLXArray.zeros([1, 1, kernel])
        self.ratio = ratio; self.kernel = kernel
    }

    /// Non-causal: replicate-pad `pad`, transposed depthwise conv × ratio, trim pad_left / pad_right.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pad = kernel / ratio - 1
        let padLeft = pad * ratio + (kernel - ratio) / 2
        let padRight = pad * ratio + (kernel - ratio + 1) / 2
        let xp = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((pad, pad)), IntOrPair((0, 0))], mode: .edge)
        // conv_transpose1d(stride r) == zero-stuff by r, pad K−1 both sides, correlate with the flipped taps
        let (b, t, c) = (xp.dim(0), xp.dim(1), xp.dim(2))
        let stuffed = concatenated([xp.expandedDimensions(axis: 2),
                                    MLXArray.zeros([b, t, ratio - 1, c], dtype: xp.dtype)], axis: 2)
            .reshaped(b, t * ratio, c)[0..., ..<((t - 1) * ratio + 1), 0...]
        let full = MLX.padded(stuffed, widths: [IntOrPair((0, 0)), IntOrPair((kernel - 1, kernel - 1)), IntOrPair((0, 0))])
        let taps = filter.reshaped(kernel)[.stride(by: -1)]
        let y = Float(ratio) * depthwise(full, taps: taps, stride: 1)
        return y[0..., padLeft ..< (y.dim(1) - padRight), 0...]
    }
}

public final class LowPassFilter1d: Module {
    @ParameterInfo(key: "filter") var filter: MLXArray
    let stride: Int, kernel: Int, causal: Bool

    public init(stride: Int, kernel: Int = 12, causal: Bool) {
        self._filter.wrappedValue = MLXArray.zeros([1, 1, kernel])
        self.stride = stride; self.kernel = kernel; self.causal = causal
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let even = kernel % 2 == 0
        let (l, r) = causal ? (kernel - 1, 0) : (kernel / 2 - (even ? 1 : 0), kernel / 2)
        let xp = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((l, r)), IntOrPair((0, 0))], mode: .edge)
        return depthwise(xp, taps: filter.reshaped(kernel), stride: stride)
    }
}

public final class DownSample1d: Module {
    @ModuleInfo(key: "lowpass") var lowpass: LowPassFilter1d

    public init(ratio: Int = 2, kernel: Int = 12, causal: Bool) {
        self._lowpass.wrappedValue = LowPassFilter1d(stride: ratio, kernel: kernel, causal: causal)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { lowpass(x) }
}

public final class Activation1d: Module, UnaryLayer {
    @ModuleInfo(key: "act") var act: SnakeBeta
    @ModuleInfo(key: "upsample") var upsample: UpSample1d
    @ModuleInfo(key: "downsample") var downsample: DownSample1d

    public init(channels: Int, logscale: Bool, causal: Bool) {
        self._act.wrappedValue = SnakeBeta(channels, logscale: logscale)
        self._upsample.wrappedValue = UpSample1d()
        self._downsample.wrappedValue = DownSample1d(causal: causal)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { downsample(act(upsample(x))) }
}

// MARK: - AMP block

public final class AMPBlock1: Module, UnaryLayer {
    @ModuleInfo(key: "convs1") var convs1: [VConv1d]
    @ModuleInfo(key: "convs2") var convs2: [VConv1d]
    @ModuleInfo(key: "activations") var activations: [Activation1d]

    public init(channels: Int, kernel: Int, dilation: [Int], causal: Bool, actCausal: Bool, logscale: Bool) {
        self._convs1.wrappedValue = dilation.map { VConv1d(channels, channels, kernel: kernel, dilation: $0, causal: causal) }
        self._convs2.wrappedValue = dilation.map { _ in VConv1d(channels, channels, kernel: kernel, dilation: 1, causal: causal) }
        self._activations.wrappedValue = (0 ..< 2 * dilation.count).map { _ in
            Activation1d(channels: channels, logscale: logscale, causal: actCausal)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for i in 0 ..< convs1.count {
            var xt = activations[2 * i](x)
            xt = convs1[i](xt)
            xt = activations[2 * i + 1](xt)
            xt = convs2[i](xt)
            x = xt + x
        }
        return x
    }
}

// MARK: - encoder

/// Conv1d_S: weight-normed nn.Conv1d with symmetric padding dilation·(k−1)//2 — key `layer.*`.
public final class Conv1d_S: Module, UnaryLayer {
    @ModuleInfo(key: "layer") var layer: VConv1d

    public init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int = 1) {
        self._layer.wrappedValue = VConv1d(inC, outC, kernel: kernel, stride: stride, padding: (kernel - 1) / 2)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { layer(x) }
}

/// Parameter-free LeakyReLU as a Module so it can sit in a keyed list.
public final class VLeakyReLU: Module, UnaryLayer {
    let slope: Float
    public init(_ slope: Float) { self.slope = slope }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { leakyRelu(x, negativeSlope: slope) }
}

/// ResStack: ModuleList of nn.Sequential(LeakyReLU(), Conv(dil), LeakyReLU(), Conv) — keys layers.N.1 / layers.N.3.
public final class ResStack: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [[UnaryLayer]]

    public init(channel: Int, kernel: Int = 3, base: Int = 3, nums: Int = 4) {
        self._layers.wrappedValue = (0 ..< nums).map { i in
            let d = Int(pow(Double(base), Double(i)))
            return [
                VLeakyReLU(0.01), VConv1d(channel, channel, kernel: kernel, dilation: d, padding: d),
                VLeakyReLU(0.01), VConv1d(channel, channel, kernel: kernel, dilation: 1, padding: 1),
            ]
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for unit in layers {
            var h = x
            for l in unit { h = l(h) }
            x = x + h
        }
        return x
    }
}

public final class Encoder: Module, UnaryLayer {
    /// nn.Sequential `generator`: [Conv1d_S, LReLU(.2)] + 6 × [Conv1d_S(stride f), ResStack, LReLU(.2)] + [Conv1d_S].
    @ModuleInfo(key: "generator") var generator: [UnaryLayer]

    public init(outChannels: Int, useVAE: Bool, downSampleFactors: [Int], channels: [Int],
                inChannels: Int = 1, baseChannels: Int = 12, projKernel: Int = 3, stackKernel: Int = 3,
                stackDilationBase: Int = 2, stacks: Int = 6) {
        let out = useVAE ? outChannels * 2 : outChannels
        var layers: [UnaryLayer] = [Conv1d_S(inChannels, baseChannels, kernel: projKernel), VLeakyReLU(0.2)]
        for (i, f) in downSampleFactors.enumerated() {
            layers += [
                Conv1d_S(channels[i], channels[i + 1], kernel: f * 2, stride: f),
                ResStack(channel: channels[i + 1], kernel: stackKernel, base: stackDilationBase, nums: stacks),
                VLeakyReLU(0.2),
            ]
        }
        layers.append(Conv1d_S(channels.last!, out, kernel: projKernel))
        self._generator.wrappedValue = layers
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for l in generator { h = l(h) }
        return h
    }
}

// MARK: - VAE

public final class BigVGANFlowVAE: Module {
    public let config: BigVGANFlowVAEConfig
    public var hopSize: Int { config.downsample_rates.reduce(1, *) }

    @ParameterInfo(key: "global_mean") var globalMean: MLXArray
    @ParameterInfo(key: "global_log_std") var globalLogStd: MLXArray
    @ModuleInfo(key: "audio_encoder") var audioEncoder: Encoder
    @ModuleInfo(key: "conv_pre") var convPre: VConv1d
    /// ups[i] is a one-element ModuleList → key ups.i.0.
    @ModuleInfo(key: "ups") var ups: [[VConvTranspose1d]]
    @ModuleInfo(key: "resblocks") var resblocks: [AMPBlock1]
    @ModuleInfo(key: "activation_post") var activationPost: Activation1d
    @ModuleInfo(key: "conv_post") var convPost: VConv1d

    public init(_ h: BigVGANFlowVAEConfig) {
        self.config = h
        self._globalMean.wrappedValue = MLXArray.zeros([h.latent_dim])
        self._globalLogStd.wrappedValue = MLXArray.ones([h.latent_dim])
        self._audioEncoder.wrappedValue = Encoder(
            outChannels: h.latent_dim, useVAE: h.use_vae, downSampleFactors: h.downsample_rates, channels: h.downsample_channels)
        self._convPre.wrappedValue = VConv1d(h.latent_dim, h.upsample_initial_channel, kernel: 7, causal: false)
        var ups = [[VConvTranspose1d]]()
        for (i, (u, k)) in zip(h.upsample_rates, h.upsample_kernel_sizes).enumerated() {
            ups.append([VConvTranspose1d(h.upsample_initial_channel >> i, h.upsample_initial_channel >> (i + 1),
                                         kernel: k, stride: u, causal: h.causal)])
        }
        self._ups.wrappedValue = ups
        var res = [AMPBlock1]()
        var ch = h.upsample_initial_channel
        for i in 0 ..< ups.count {
            ch = h.upsample_initial_channel >> (i + 1)
            for (k, d) in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes) {
                res.append(AMPBlock1(channels: ch, kernel: k, dilation: d, causal: h.causal, actCausal: h.act_causal,
                                     logscale: h.snake_logscale))
            }
        }
        self._resblocks.wrappedValue = res
        self._activationPost.wrappedValue = Activation1d(channels: ch, logscale: h.snake_logscale, causal: h.act_causal)
        self._convPost.wrappedValue = VConv1d(ch, 1, kernel: 7, causal: h.causal, bias: false)
    }

    /// Posterior statistics: sample (B, T, 1) → (mean, log_std), each (B, T/hop, D).
    public func encodeStats(_ sample: MLXArray) -> (MLXArray, MLXArray) {
        let s = split(audioEncoder(sample), parts: 2, axis: -1)
        return (s[0], s[1])
    }

    /// `encoding_and_normalization` with the reference noise supplied: (mean + noise·exp(log_std) − μ_g) / √σ_g.
    /// `noise` nil ⇒ the posterior mean (the deterministic gate path). Output (B, T/hop, D) fp32.
    public func encodingAndNormalization(_ sample: MLXArray, noise: MLXArray?) -> MLXArray {
        let (mean, logStd) = encodeStats(sample)
        var z = mean.asType(.float32)
        if let noise { z = z + noise.asType(.float32) * exp(logStd.asType(.float32)) }
        return (z - globalMean.asType(.float32)) / sqrt(globalLogStd.asType(.float32))
    }

    public func denormalize(_ latents: MLXArray) -> MLXArray {
        latents.asType(.float32) * sqrt(globalLogStd.asType(.float32)) + globalMean.asType(.float32)
    }

    /// latents (B, T, D) → waveform (B, T·hop, 1), clamped to ±1.
    public func inferenceFromLatents(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(-1) == config.latent_dim)
        var x = convPre(x)
        let nk = config.resblock_kernel_sizes.count
        for i in 0 ..< ups.count {
            for up in ups[i] { x = up(x) }
            var xs: MLXArray? = nil
            for j in 0 ..< nk {
                let r = resblocks[i * nk + j](x)
                xs = xs.map { $0 + r } ?? r
            }
            x = xs! / Float(nk)
        }
        x = activationPost(x)
        x = convPost(x)
        return clip(x, min: -1.0, max: 1.0)
    }
}
