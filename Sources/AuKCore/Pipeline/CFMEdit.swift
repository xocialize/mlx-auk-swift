// CFMEdit.swift — auk/model/cfm_edit.py, inference half: the ELMo-style fusion of the encoder's hidden states
// (`encode_text`) and the Euler flow sampler (`sample`). Keys: `transformer.*`, `layer_weights`, `layer_scale` — the
// whole of dit.safetensors.
//
// Sampler, as upstream: the ODE state is the target only (the reference is prepended inside the backbone);
// t = linspace(0, 1, steps + 1) bent by sway (t + s·(cos(πt/2) − 1 + t)), or an explicit t_grid (AuK-Flash);
// torchdiffeq's fixed-grid Euler is x += (t[i+1] − t[i]) · v(t[i], x); CFG doubles the batch inside the backbone and
// combines v_c + (v_c − v_u)·w; the projected text is cached across steps.

import Foundation
import MLX
import MLXNN

/// What a semantic encoder hands the DiT. The DiT was trained against one encoder's hidden-state stack, so a new
/// encoder means new DiT weights — this seam exists for the permissive encoder Tencent has said it will release.
public protocol AukConditionEncoder: AnyObject {
    /// Number of transformer layers whose outputs are fused (hidden_states[1...numLayers]).
    var numLayers: Int { get }
    /// SPDX id (or LicenseRef-) of this encoder's weights.
    var weightLicense: String { get }
    /// Runs the encoder on one prompt and visits hidden states 0…numLayers in HF order. Returns the token count.
    func forEachHiddenState(_ prompt: AukPrompt, visit: (Int, MLXArray) -> Void) throws -> Int
}

public struct SamplingConfig: Codable, Sendable {
    public var nfe: Int
    public var cfg_strength: Double
    public var sway_sampling_coef: Double?
    public var t_grid: [Double]?

    public init(nfe: Int, cfg_strength: Double, sway_sampling_coef: Double?, t_grid: [Double]?) {
        self.nfe = nfe; self.cfg_strength = cfg_strength; self.sway_sampling_coef = sway_sampling_coef; self.t_grid = t_grid
    }

    /// The time grid the sampler integrates over (fp32, as torch builds it).
    public func timeGrid() -> [Float] {
        if let g = t_grid { return g.map(Float.init) }
        var t = (0 ... nfe).map { Float($0) / Float(nfe) }
        if let s = sway_sampling_coef {
            let sf = Float(s)
            t = t.map { $0 + sf * (cos(Float.pi / 2 * $0) - 1 + $0) }
        }
        return t
    }
}

public final class CFMEdit: Module {
    @ModuleInfo(key: "transformer") public var transformer: Flux2Edit
    @ParameterInfo(key: "layer_weights") var layerWeights: MLXArray
    @ParameterInfo(key: "layer_scale") var layerScale: MLXArray

    public init(_ c: Flux2EditConfig, numEncoderLayers: Int) {
        self._transformer.wrappedValue = Flux2Edit(c)
        self._layerWeights.wrappedValue = MLXArray.zeros([numEncoderLayers])
        self._layerScale.wrappedValue = MLXArray.ones([1])
    }

    /// Fused condition (1, nt, d_llm) fp32: Σ_i softmax(layer_weights)_i · LN(h_i) · layer_scale over i = 1…L, where LN
    /// is F.layer_norm without affine (eps 1e-5). Accumulated as the encoder streams, so no (L, nt, d) stack exists.
    public func encodeText(_ prompt: AukPrompt, encoder: AukConditionEncoder,
                           capture: ((Int, MLXArray) -> Void)? = nil) throws -> (MLXArray, MLXArray) {
        let w = softmax(layerWeights.asType(.float32), axis: 0)
        var acc: MLXArray? = nil
        let n = try encoder.forEachHiddenState(prompt) { i, h in
            capture?(i, h)
            guard i >= 1 else { return }
            let term = MLXFast.layerNorm(h.asType(.float32), weight: nil, bias: nil, eps: 1e-5) * w[i - 1]
            acc = acc.map { $0 + term } ?? term
            eval(acc!)
        }
        guard let acc else { throw AuKError.invalidInput("encoder produced no hidden states") }
        let fused = acc * layerScale.asType(.float32)
        return (fused, MLXArray.ones([1, n]).asType(.bool))
    }

    /// The flow sampler. `cond` is the normalised reference latent (1, Np, D) — Np may be 0 (instruct TTS) — of which
    /// the first `refLen` frames are valid; `y0` the starting noise (1, N, D). Returns the generated target latent (1, N, D), still normalised.
    public func sample(cond: MLXArray, refLen: Int? = nil, textEmbeds: MLXArray, contextMask: MLXArray, y0: MLXArray,
                       sampling: SamplingConfig, dtype: DType,
                       onStep: ((Int, Int) -> Void)? = nil,
                       capture: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        // upstream keeps every encoder frame and masks the ones past len // hop (`lens_to_mask(ref_lens, length=Np)`):
        // the extra frame stays in the sequence (so it shifts the target's RoPE positions) but is key-masked
        let np = cond.dim(1), valid = min(refLen ?? np, np)
        let refMask = MLXArray(0 ..< np).expandedDimensions(axis: 0) .< MLXArray(Int32(valid))
        let ref: MLXArray? = cond.dim(1) > 0 ? cond.asType(dtype) : nil
        let text = textEmbeds.asType(dtype)
        let w = Float(sampling.cfg_strength)
        let t = sampling.timeGrid()
        var x = y0.asType(.float32)
        transformer.clearCache()
        defer { transformer.clearCache() }

        for i in 0 ..< (t.count - 1) {
            let ti = MLXArray(t[i])
            let v: MLXArray
            let firstCapture: ((String, MLXArray) -> Void)? = i == 0 ? capture : nil
            if w < 1e-5 {
                v = transformer(x: x.asType(dtype), text: text, time: ti, mask: nil, cMask: contextMask,
                                ref: ref, refMask: refMask, cache: true, capture: firstCapture).asType(.float32)
                firstCapture?("raw", v)
            } else {
                let pred = transformer(x: x.asType(dtype), text: text, time: ti, mask: nil, cMask: contextMask,
                                       ref: ref, refMask: refMask, cfgInfer: true, cache: true,
                                       capture: firstCapture).asType(.float32)
                firstCapture?("raw", pred)
                let parts = split(pred, parts: 2, axis: 0)
                v = parts[0] + (parts[0] - parts[1]) * w
            }
            if i == 0 { capture?("out", v) }
            x = x + (t[i + 1] - t[i]) * v
            eval(x)
            onStep?(i + 1, t.count - 1)
        }
        return x
    }
}
