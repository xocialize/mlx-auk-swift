// auk-gates — parity gates against the fp32 torch oracle (WIP/auk-port/oracle/dump_goldens.py), PORTING-SPEC §4.
//
//   auk-gates vae     <golden> <model>            encoder stats, normalised latents (injected noise), decoder
//   auk-gates dit     <golden> <model>            one backbone call on the oracle's call-0 inputs, per block
//   auk-gates encoder <golden> <model> <encoder>  token ids, mel features, audio tower, hidden states, fused condition
//   auk-gates e2e     <golden> <model> <encoder> [out.wav]   whole request with the oracle's noise
//   auk-gates render  <model> <encoder> "<instruction>" <in.wav|-> <seconds|-> <out.wav> [seed]
//
// Options (before the subcommand): --cpu (fp32 reference lane on the CPU stream), --dtype float32|bfloat16,
// --enc-dtype float32|bfloat16|… (encoder only).

import AuKCore
import Foundation
import MLX
import MLXNN

// MARK: - options

var args = Array(CommandLine.arguments.dropFirst())
var useCPU = false
var dtype: DType = .float32
var encDType: DType? = nil
func parseDType(_ s: String) -> DType {
    switch s { case "bfloat16", "bf16": return .bfloat16; case "float16", "fp16": return .float16; default: return .float32 }
}
while let a = args.first, a.hasPrefix("--") {
    args.removeFirst()
    switch a {
    case "--cpu": useCPU = true
    case "--dtype": dtype = parseDType(args.removeFirst())
    case "--enc-dtype": encDType = parseDType(args.removeFirst())
    default: fatalError("unknown option \(a)")
    }
}
guard let cmd = args.first else {
    print("usage: auk-gates [--cpu] [--dtype d] [--enc-dtype d] vae|dit|encoder|e2e|render …"); exit(2)
}
let rest = Array(args.dropFirst())

// MARK: - metrics

struct Row { let name: String; let maxAbs: Float; let relL2: Float; let cos: Float; let shape: [Int] }
var rows = [Row]()

func compare(_ name: String, _ got: MLXArray, _ ref: MLXArray) {
    let g = got.asType(.float32), r = ref.asType(.float32)
    guard g.shape == r.shape else {
        print(String(format: "  %-28@ SHAPE MISMATCH got %@ ref %@", name as NSString, "\(g.shape)" as NSString, "\(r.shape)" as NSString))
        rows.append(Row(name: name, maxAbs: .infinity, relL2: .infinity, cos: 0, shape: g.shape)); return
    }
    let d = g - r
    let maxAbs = abs(d).max().item(Float.self)
    let relL2 = (sqrt((d * d).sum()) / maximum(sqrt((r * r).sum()), MLXArray(Float(1e-30)))).item(Float.self)
    let cos = ((g * r).sum() / maximum(sqrt((g * g).sum()) * sqrt((r * r).sum()), MLXArray(Float(1e-30)))).item(Float.self)
    rows.append(Row(name: name, maxAbs: maxAbs, relL2: relL2, cos: cos, shape: g.shape))
    print(String(format: "  %-28@ max|Δ| %.3e   relL2 %.3e   cos %.7f   %@", name as NSString, maxAbs, relL2, cos, "\(g.shape)" as NSString))
}

func npy(_ dir: URL, _ name: String) throws -> MLXArray { try NPY.load(dir.appendingPathComponent("\(name).npy")) }

struct Meta: Decodable {
    let instruction: String
    let audio: String?
    let gen_seconds: Double?
    let gen_latent_len: Int
}

func loadMeta(_ dir: URL) throws -> Meta {
    try JSONDecoder().decode(Meta.self, from: Data(contentsOf: dir.appendingPathComponent("meta.json")))
}

/// (1, C, T) channels-first golden → (1, T, C).
func cl(_ a: MLXArray) -> MLXArray { a.transposed(0, 2, 1) }

// MARK: - lanes

func vaeGate(_ golden: URL, _ modelDir: URL) throws {
    let (_, _, vae) = try AukPipeline.loadModel(modelDir, dtype: dtype)
    if FileManager.default.fileExists(atPath: golden.appendingPathComponent("ref24k.npy").path) {
        let ref24 = cl(try npy(golden, "ref24k"))                         // (1, T, 1)
        let (mean, logStd) = vae.encodeStats(ref24.asType(dtype))
        compare("enc_mean", mean, cl(try npy(golden, "enc_mean")))
        compare("enc_log_std", logStd, cl(try npy(golden, "enc_log_std")))
        let lat = vae.encodingAndNormalization(ref24.asType(dtype), noise: cl(try npy(golden, "enc_noise")))
        compare("ref_latents", lat, try npy(golden, "ref_latents"))
    } else {
        print("  (no reference audio in this task — encoder lane skipped)")
    }
    let decIn = cl(try npy(golden, "dec_in"))                              // (1, T, 64), denormalised
    let wave = vae.inferenceFromLatents(decIn.asType(dtype))
    compare("audio_out", wave, cl(try npy(golden, "audio_out")))
}

func ditGate(_ golden: URL, _ modelDir: URL) throws {
    let (_, model, _) = try AukPipeline.loadModel(modelDir, dtype: dtype)
    let x = try npy(golden, "call0_x"), text = try npy(golden, "call0_text")
    let refURL = golden.appendingPathComponent("call0_ref.npy")
    let ref: MLXArray? = FileManager.default.fileExists(atPath: refURL.path) ? try npy(golden, "call0_ref") : nil
    let cMask = try npy(golden, "call0_c_mask").asType(.bool)
    let refMask = FileManager.default.fileExists(atPath: golden.appendingPathComponent("call0_ref_mask.npy").path)
        ? try npy(golden, "call0_ref_mask").asType(.bool) : nil
    let out0 = try npy(golden, "call0_out")
    let cfg = out0.dim(0) == 2 * x.dim(0)
    let t = try npy(golden, "call0_time").reshaped([])
    var captured = [String: MLXArray]()
    let out = model.transformer(
        x: x.asType(dtype), text: text.asType(dtype), time: t, mask: nil, cMask: cMask, ref: ref?.asType(dtype),
        refMask: refMask, cfgInfer: cfg, cache: false, capture: { captured[$0] = $1 })
    for i in [0, 4, 9] {
        compare("block\(i)_c", captured["block\(i)_c"]!, try npy(golden, "call0_block\(i)_c"))
        compare("block\(i)_x", captured["block\(i)_x"]!, try npy(golden, "call0_block\(i)_x"))
    }
    for i in [0, 9, 19] { compare("single\(i)", captured["single\(i)"]!, try npy(golden, "call0_single\(i)")) }
    compare("velocity", out, out0)
}

func encoderGate(_ golden: URL, _ modelDir: URL, _ encDir: URL) async throws {
    let (_, model, _) = try AukPipeline.loadModel(modelDir, dtype: dtype)
    let enc = try await QwenOmniConditionEncoder(directory: encDir, dtype: encDType ?? dtype)
    let meta = try loadMeta(golden)
    let goldIds = try npy(golden, "input_ids").reshaped(-1).asArray(Int32.self)

    var featsOverride: MLXArray? = nil
    var audio16k: [Float]? = nil
    if meta.audio != nil {
        audio16k = try npy(golden, "audio16k").asArray(Float.self)
        let fam = try npy(golden, "feature_attention_mask").reshaped(-1)
        let nValid = fam.sum().item(Int.self)
        let goldFeats = try npy(golden, "input_features")[0, 0..., ..<nValid].transposed(1, 0)   // (L, 128)
        let mine = enc.features(audio16k!)
        compare("input_features(audio16k)", mine, goldFeats)
        featsOverride = goldFeats
        let tower = enc.audioEmbeds(features: goldFeats)
        compare("audio_tower_out", tower, try npy(golden, "audio_tower_out"))
    }
    let prompt = AukPrompt(instruction: meta.instruction, audio16k: audio16k)
    var hidden = [Int: MLXArray]()
    let keep: Set<Int> = [0, 1, 18, 35, 36]
    let w = softmax(try npy(golden.deletingLastPathComponent(), "layer_weights"), axis: 0)
    var acc: MLXArray? = nil
    _ = try enc.forEachHiddenState(prompt, featuresOverride: featsOverride) { i, h in
        if keep.contains(i) { hidden[i] = h; eval(h) }
        if i >= 1 {
            let term = MLXFast.layerNorm(h.asType(.float32), weight: nil, bias: nil, eps: 1e-5) * w[i - 1]
            acc = acc.map { $0 + term } ?? term
            eval(acc!)
        }
    }
    let ids = enc.lastInputIds
    let idsMatch = ids == goldIds
    print("  input_ids: \(ids.count) tokens vs golden \(goldIds.count) — \(idsMatch ? "IDENTICAL" : "DIFFER")")
    if !idsMatch {
        let firstDiff = zip(ids, goldIds).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(ids.count, goldIds.count)
        print("    first difference at \(firstDiff): got \(Array(ids[max(0, firstDiff - 3) ..< min(ids.count, firstDiff + 5)])) golden \(Array(goldIds[max(0, firstDiff - 3) ..< min(goldIds.count, firstDiff + 5)]))")
    }
    for i in keep.sorted() { compare("hidden_\(i)", hidden[i]!, try npy(golden, "hidden_\(i)")) }
    let scale = try npy(golden.deletingLastPathComponent(), "layer_scale")
    compare("fused", acc! * scale, try npy(golden, "fused"))
    _ = model
}

func e2eGate(_ golden: URL, _ modelDir: URL, _ encDir: URL, _ out: URL?) async throws {
    let pipe = try await AukPipeline.load(modelDir: modelDir, encoderDir: encDir, dtype: dtype, encoderDType: encDType)
    let meta = try loadMeta(golden)
    var audio: [Float]? = nil
    var rate = 24000
    var vaeNoise: MLXArray? = nil
    var feats: MLXArray? = nil
    if meta.audio != nil {
        // feed the oracle's own 24 kHz reference and 16 kHz features so this lane isolates the model from resampling
        audio = try npy(golden, "ref24k").reshaped(-1).asArray(Float.self)
        rate = 24000
        vaeNoise = cl(try npy(golden, "enc_noise"))
        let nValid = try npy(golden, "feature_attention_mask").sum().item(Int.self)
        feats = try npy(golden, "input_features")[0, 0..., ..<nValid].transposed(1, 0)
    }
    let y0 = try npy(golden, "y0").expandedDimensions(axis: 0)
    var cap = [String: MLXArray]()
    let t0 = Date()
    let wave = try pipe.generate(
        AukRequest(instruction: meta.instruction.replacingOccurrences(of: "|<no_prompt_audio>|", with: ""),
                   audio: audio, sampleRate: rate, genSeconds: meta.gen_seconds),
        inject: AukInjectedNoise(y0: y0, vaeNoise: vaeNoise, features: feats),
        capture: { cap[$0] = $1 })
    let wall = Date().timeIntervalSince(t0)
    if let r = cap["ref_latents"] { compare("ref_latents", r, try npy(golden, "ref_latents")) }
    compare("fused", cap["fused"]!, try npy(golden, "fused"))
    if let raw = cap["raw"] { compare("call0_out", raw, try npy(golden, "call0_out")) }
    compare("gen_latent", cap["gen_latent"]!, try npy(golden, "gen_latent"))
    compare("audio_out", cap["audio_out"]!, cl(try npy(golden, "audio_out")))
    let secs = Double(wave.count) / 24000
    print(String(format: "  wall %.2f s for %.2f s of audio (RTF %.3f)", wall, secs, wall / secs))
    if let out { try WAV.write(wave, sampleRate: 24000, to: out); print("  wrote \(out.path)") }
}

func render(_ modelDir: URL, _ encDir: URL, _ instruction: String, _ input: String, _ seconds: String, _ out: URL,
            seed: UInt64) async throws {
    let pipe = try await AukPipeline.load(modelDir: modelDir, encoderDir: encDir, dtype: dtype, encoderDType: encDType)
    var audio: [Float]? = nil
    var rate = 24000
    if input != "-" { (audio, rate) = try WAV.read(URL(fileURLWithPath: input)) }
    let t0 = Date()
    let wave = try pipe.generate(AukRequest(instruction: instruction, audio: audio, sampleRate: rate,
                                            genSeconds: seconds == "-" ? nil : Double(seconds), seed: seed))
    let wall = Date().timeIntervalSince(t0)
    try WAV.write(wave, sampleRate: 24000, to: out)
    let secs = Double(wave.count) / 24000
    print(String(format: "wrote %@ — %.2f s audio in %.2f s (RTF %.3f), peak %.2f GB", out.path, secs, wall, wall / secs,
                 Double(Memory.peakMemory) / 1e9))
}

// MARK: - dispatch

func run() async throws {
    switch cmd {
    case "vae": try vaeGate(URL(fileURLWithPath: rest[0]), URL(fileURLWithPath: rest[1]))
    case "dit": try ditGate(URL(fileURLWithPath: rest[0]), URL(fileURLWithPath: rest[1]))
    case "encoder":
        try await encoderGate(URL(fileURLWithPath: rest[0]), URL(fileURLWithPath: rest[1]), URL(fileURLWithPath: rest[2]))
    case "e2e":
        try await e2eGate(URL(fileURLWithPath: rest[0]), URL(fileURLWithPath: rest[1]), URL(fileURLWithPath: rest[2]),
                          rest.count > 3 ? URL(fileURLWithPath: rest[3]) : nil)
    case "render":
        try await render(URL(fileURLWithPath: rest[0]), URL(fileURLWithPath: rest[1]), rest[2], rest[3], rest[4],
                         URL(fileURLWithPath: rest[5]), seed: rest.count > 6 ? UInt64(rest[6]) ?? 0 : 0)
    default: print("unknown command \(cmd)"); exit(2)
    }
}

let sema = DispatchSemaphore(value: 0)
Task {
    do {
        if useCPU { try await Device.withDefaultDevice(.cpu) { try await run() } } else { try await run() }
    } catch {
        print("ERROR: \(error)")
        exit(1)
    }
    sema.signal()
}
sema.wait()
