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
import Darwin
import Foundation
import MLXAuK
import MLXServeCore
import MLXToolKit
import MLX
import MLXNN

// MARK: - options

var args = Array(CommandLine.arguments.dropFirst())
var useCPU = false
var dtype: DType = .float32
var encDType: DType? = nil
var ditBits: Int? = nil
var encBits: Int? = nil
var seeds: [UInt64] = [0, 1, 2]
func parseDType(_ s: String) -> DType {
    switch s { case "bfloat16", "bf16": return .bfloat16; case "float16", "fp16": return .float16; default: return .float32 }
}
while let a = args.first, a.hasPrefix("--") {
    args.removeFirst()
    switch a {
    case "--cpu": useCPU = true
    case "--dtype": dtype = parseDType(args.removeFirst())
    case "--enc-dtype": encDType = parseDType(args.removeFirst())
    case "--dit-bits": ditBits = Int(args.removeFirst())
    case "--enc-bits": encBits = Int(args.removeFirst())
    case "--seeds": seeds = args.removeFirst().split(separator: ",").compactMap { UInt64($0) }
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

/// V2b: every golden task × seeds through the production path (own noise, own resampling, own features).
func matrix(_ goldenVariant: URL, _ assets: URL, _ modelDir: URL, _ encDir: URL, _ outDir: URL) async throws {
    let q = AukQuantization(ditBits: ditBits, encoderBits: encBits)
    let pipe = try await AukPipeline.load(modelDir: modelDir, encoderDir: encDir, dtype: dtype, encoderDType: encDType,
                                          quantization: q)
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let log = outDir.appendingPathComponent("renders.jsonl")
    if !FileManager.default.fileExists(atPath: log.path) { FileManager.default.createFile(atPath: log.path, contents: nil) }
    let fh = try FileHandle(forWritingTo: log); fh.seekToEndOfFile()
    let tasks = try FileManager.default.contentsOfDirectory(atPath: goldenVariant.path).filter {
        FileManager.default.fileExists(atPath: goldenVariant.appendingPathComponent("\($0)/meta.json").path)
    }.sorted()
    for task in tasks {
        let meta = try loadMeta(goldenVariant.appendingPathComponent(task))
        var audio: [Float]? = nil
        var rate = 24000
        if let a = meta.audio { (audio, rate) = try WAV.read(assets.appendingPathComponent(a)) }
        let instruction = meta.instruction.replacingOccurrences(of: "|<no_prompt_audio>|", with: "")
        for seed in seeds {
            let out = outDir.appendingPathComponent("\(task)-s\(seed).wav")
            if FileManager.default.fileExists(atPath: out.path) { continue }
            Memory.peakMemory = 0
            let t0 = Date()
            let wave = try pipe.generate(AukRequest(instruction: instruction, audio: audio, sampleRate: rate,
                                                    genSeconds: meta.gen_seconds, seed: seed))
            let wall = Date().timeIntervalSince(t0)
            try WAV.write(wave, sampleRate: 24000, to: out)
            let rec: [String: Any] = ["task": task, "seed": seed, "path": out.path, "wall_s": wall,
                                      "audio_s": Double(wave.count) / 24000, "peak_gb": Double(Memory.peakMemory) / 1e9]
            fh.write(try JSONSerialization.data(withJSONObject: rec)); fh.write("\n".data(using: .utf8)!)
            print(String(format: "  %@ s%d  %.2f s (RTF %.3f) peak %.2f GB", task as NSString, Int(seed), wall,
                         wall / (Double(wave.count) / 24000), Double(Memory.peakMemory) / 1e9))
        }
    }
    try fh.close()
}

/// Writes a pre-quantised encoder bundle from the bf16 one: layer Linears → affine `bits` (group 64), everything else
/// as stored; config.json gains `quantization`; tokenizer / processor files copied verbatim.
func quantizeEncoder(_ src: URL, _ dst: URL, bits: Int) throws {
    let fm = FileManager.default
    try fm.createDirectory(at: dst, withIntermediateDirectories: true)
    var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: src.appendingPathComponent("config.json"))) as! [String: Any]
    let config = try ConfigIO.load(QwenOmniThinkerConfig.self, from: src.appendingPathComponent("config.json"))
    let thinker = Qwen2_5OmniThinker(config)
    let w = try WeightIO.load(src.appendingPathComponent("encoder.safetensors"), dtype: .bfloat16)
    try WeightIO.apply(WeightIO.scoped(w, prefix: "thinker."), to: thinker, component: "thinker")
    thinker.quantizeLayers(groupSize: 64, bits: bits)
    var out = [String: MLXArray]()
    for (k, v) in thinker.parameters().flattened() { out["thinker." + k] = v }
    eval(Array(out.values))
    try save(arrays: out, metadata: ["format": "mlx"], url: dst.appendingPathComponent("encoder.safetensors"))
    cfg["quantization"] = ["group_size": 64, "bits": bits]
    cfg["dtype"] = "bfloat16"
    try JSONSerialization.data(withJSONObject: cfg, options: [.prettyPrinted, .sortedKeys])
        .write(to: dst.appendingPathComponent("config.json"))
    for f in try fm.contentsOfDirectory(atPath: src.path) where f != "encoder.safetensors" && f != "config.json" {
        let to = dst.appendingPathComponent(f)
        if fm.fileExists(atPath: to.path) { try fm.removeItem(at: to) }
        try fm.copyItem(at: src.appendingPathComponent(f), to: to)
    }
    let bytes = (try? fm.attributesOfItem(atPath: dst.appendingPathComponent("encoder.safetensors").path)[.size] as? Int) ?? 0
    print(String(format: "wrote %@ — %d tensors, %.2f GB", dst.path, out.count, Double(bytes) / 1e9))
}

func physFootprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : .nan
}

/// The consumer path: MLXServeEngine register → prepare → run → evict, phys_footprint at each step (the manifest's
/// C-memory numbers). The default .permissiveOnly engine must REFUSE the package; the .any engine (the operator's
/// --allow-research acceptance) admits it.
func validate(variant: String, quant q: String, _ modelDir: URL, _ encDir: URL, _ assets: URL, _ outDir: URL) async throws {
    let quant: Quant = q == "int8" ? .int8 : q == "int4" ? .int4 : .bf16
    let v: AuKVariant = variant == "base" ? .base : .flash
    let config = AuKConfiguration(variant: v, quant: quant, modelDirectory: modelDir, encoderDirectory: encDir)
    let registration: PackageRegistration = v == .base ? AuKPackage.registration : AuKFlashPackage.registration
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    let strict = MLXServeEngine()
    do {
        _ = try await strict.register(registration, configuration: config, id: PackageID("auk-strict"))
        let advisories = await strict.licenseAdvisories
        print("  [LIC] .permissiveOnly registered with \(advisories.count) advisories — \(advisories.isEmpty ? "NOT REFUSED (FAIL)" : "advisory only")")
    } catch {
        print("  [LIC] .permissiveOnly refused registration: \(String(describing: error).prefix(160))")
    }

    let engine = MLXServeEngine(policy: .any)
    let id = try await engine.register(registration, configuration: config, id: PackageID(v == .base ? "auk" : "auk-flash"))
    let before = physFootprintGB()
    let t0 = Date()
    try await engine.prepare(.tts, package: id)
    let afterLoad = physFootprintGB()
    print(String(format: "  [VAL] %@ %@ prepared in %.1fs (warm-up incl.); phys %.2f → %.2f GB (resident +%.2f)",
                 variant, q, Date().timeIntervalSince(t0), before, afterLoad, afterLoad - before))
    var peak = afterLoad
    func wav(_ rel: String) throws -> Audio {
        let (s, r) = try WAV.read(assets.appendingPathComponent(rel))
        var d = Data(); func u32(_ x: UInt32) { withUnsafeBytes(of: x.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ x: UInt16) { withUnsafeBytes(of: x.littleEndian) { d.append(contentsOf: $0) } }
        d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + s.count * 2)); d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1); u32(UInt32(r)); u32(UInt32(r * 2)); u16(2); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(UInt32(s.count * 2))
        for x in s { withUnsafeBytes(of: Int16(max(-1, min(1, x)) * 32767).littleEndian) { d.append(contentsOf: $0) } }
        return Audio(format: .wav, data: d, sampleRate: r, channels: 1)
    }
    let cases: [(String, Capability, any CapabilityRequest)] = [
        ("tts-clone", .tts, TTSRequest(text: "Ladies and gentlemen, it's an honor to address such a distinguished audience.",
                                       voice: VoiceSelector(.referenceAudio(try wav("zero-shot-tts/ref.wav"))), targetDuration: 5.5,
                                       metaData: ["seed": .int(7)])),
        ("tts-design", .tts, TTSRequest(text: "Welcome home, how was work today?",
                                        emotion: .textDescription("A young woman, warm and gentle, speaking softly."),
                                        metaData: ["seed": .int(7)])),
        ("edit-happy", .speechEdit, SpeechEditRequest(audio: try wav("emotion-edit/en-1-input.wav"), transcript: "", edit: .emotion("happy"), seed: 7)),
        ("edit-whisper", .speechEdit, SpeechEditRequest(audio: try wav("emotion-edit/en-1-input.wav"), transcript: "", edit: .style("whisper"), seed: 7)),
        ("edit-denoise", .speechEdit, SpeechEditRequest(audio: try wav("se/se-zh-1-input.wav"), transcript: "", edit: .denoise, seed: 7)),
    ]
    for (name, cap, req) in cases {
        if cap == .speechEdit { try await engine.prepare(.speechEdit, package: id) }
        let t = Date()
        let resp = try await engine.run(req, package: id)
        let audio = (resp as? TTSResponse)?.audio ?? (resp as! SpeechEditResponse).audio
        let samples = audio.data.dropFirst(44).withUnsafeBytes { raw -> [Float] in raw.bindMemory(to: Int16.self).map { Float($0) / 32768 } }
        let secs = Double(samples.count) / 24000
        let rms = 20 * log10(max(sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(max(samples.count, 1))), 1e-9))
        let foot = physFootprintGB(); peak = max(peak, foot)
        let wall = Date().timeIntervalSince(t)
        try audio.data.write(to: outDir.appendingPathComponent("\(variant)-\(q)-\(name).wav"))
        print(String(format: "  [RUN] %@: %.2fs audio, %.1f dBFS, %.2fs wall (RTF %.2f), phys %.2f GB", name, secs, rms, wall, wall / max(secs, 0.01), foot))
    }
    do {
        _ = try await engine.run(SpeechEditRequest(audio: Audio(format: .wav, data: Data([1, 2, 3]), sampleRate: 24000, channels: 1),
                                                   transcript: "", edit: .emotion("furious")), package: id)
        print("  [REQ] FAIL: undeclared emotion admitted")
    } catch { print("  [REQ] undeclared emotion refused: \(String(describing: error).prefix(100))") }
    await engine.evict(package: id)
    var afterEvict = physFootprintGB()
    let relieved = malloc_zone_pressure_relief(nil, 0)
    let afterRelief = physFootprintGB()
    print(String(format: "  [VAL] after evict %.2f GB; malloc pressure relief released %.2f GB → %.2f GB", afterEvict,
                 Double(relieved) / 1e9, afterRelief))
    afterEvict = afterRelief
    // a second residency cycle: residue that is reused keeps the second peak near the first; a leak stacks
    try await engine.prepare(.tts, package: id)
    let reload = physFootprintGB()
    _ = try await engine.run(cases[1].2, package: id)
    await engine.evict(package: id)
    print(String(format: "  [VAL] second cycle: phys after re-prepare %.2f GB (first %.2f), after second evict %.2f GB",
                 reload, afterLoad, physFootprintGB()))
    print(String(format: "  [VAL] %@ %@: resident %.2f GB, peak phys %.2f GB (activation %.2f GB), after evict %.2f GB (MLX active %.2f, cache %.2f GB)",
                 variant, q, afterLoad - before, peak, peak - afterLoad, afterEvict,
                 Double(Memory.activeMemory) / 1e9, Double(Memory.cacheMemory) / 1e9))
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
    case "quantize-encoder":
        try quantizeEncoder(URL(fileURLWithPath: rest[0]), URL(fileURLWithPath: rest[1]), bits: Int(rest.count > 2 ? rest[2] : "8") ?? 8)
    case "validate":
        try await validate(variant: rest[0], quant: rest[1], URL(fileURLWithPath: rest[2]), URL(fileURLWithPath: rest[3]),
                           URL(fileURLWithPath: rest[4]), URL(fileURLWithPath: rest[5]))
    case "matrix":
        try await matrix(URL(fileURLWithPath: rest[0]), URL(fileURLWithPath: rest[1]), URL(fileURLWithPath: rest[2]),
                         URL(fileURLWithPath: rest[3]), URL(fileURLWithPath: rest[4]))
    default: print("unknown command \(cmd)"); exit(2)
    }
}

do {
    if useCPU { try await Device.withDefaultDevice(.cpu) { try await run() } } else { try await run() }
} catch {
    print("ERROR: \(error)")
    exit(1)
}
