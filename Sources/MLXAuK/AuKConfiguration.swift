// AuKConfiguration.swift — init-time configuration for `AuKPackage` / `AuKFlashPackage` (C9). Per-request text / audio /
// edit ride the canonical `TTSRequest` / `SpeechEditRequest`.
//
// TWO weight sources back the model:
//   • "model"   — the converted AuK checkpoint: mlx-community/AuK-<tier> or AuK-Flash-<tier> (dit + vae + config, MIT);
//   • "encoder" — the Qwen2.5-Omni-3B thinker: mlx-community/Qwen2.5-Omni-3B-Thinker-<tier> (Qwen RESEARCH licence).
// `.fp32` is the parity precision: it upcasts the bf16 bundles at load and is not a published tier.

import Foundation
import MLX
import MLXToolKit
import AuKCore

public enum AuKVariant: String, Codable, Sendable {
    case base, flash

    public var modelRepoStem: String { self == .flash ? "mlx-community/AuK-Flash" : "mlx-community/AuK" }
}

public struct AuKConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var variant: AuKVariant
    /// `.bf16` (DiT, VAE and encoder bf16), `.int8` / `.int4` (encoder 8- / 4-bit, group 64; DiT and VAE bf16 — all
    /// three pass the V2b gate, PORTING-SPEC §7), `.fp32` (parity).
    public var quant: Quant
    public var revision: String?
    public var encoderRevision: String?
    /// Explicit bundle directories (dev escape hatch — never touch the network).
    public var modelDirectory: URL?
    public var encoderDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// Run one short render at load so the first real request runs at the steady RTF (Metal kernel compile).
    public var warmUp: Bool

    public init(variant: AuKVariant = .flash, quant: Quant = .bf16, revision: String? = nil, encoderRevision: String? = nil,
                modelDirectory: URL? = nil, encoderDirectory: URL? = nil, modelsRootDirectory: URL? = nil,
                warmUp: Bool = true) {
        self.variant = variant; self.quant = quant; self.revision = revision; self.encoderRevision = encoderRevision
        self.modelDirectory = modelDirectory; self.encoderDirectory = encoderDirectory
        self.modelsRootDirectory = modelsRootDirectory; self.warmUp = warmUp
    }

    /// The DiT + VAE bundle: published bf16 only (the DiT is not quantised in any shipped tier).
    public var modelRepo: String { variant.modelRepoStem + "-bf16" }
    /// The encoder bundle for the tier; fp32 upcasts the bf16 one.
    public var encoderRepo: String {
        "mlx-community/Qwen2.5-Omni-3B-Thinker-" + (quant == .int8 ? "8bit" : quant == .int4 ? "4bit" : "bf16")
    }

    var dtype: DType { quant == .fp32 ? .float32 : .bfloat16 }

    private enum CodingKeys: String, CodingKey { case variant, quant, revision, encoderRevision, warmUp }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        variant = try c.decode(AuKVariant.self, forKey: .variant)
        quant = try c.decode(Quant.self, forKey: .quant)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        encoderRevision = try c.decodeIfPresent(String.self, forKey: .encoderRevision)
        warmUp = try c.decodeIfPresent(Bool.self, forKey: .warmUp) ?? true
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension AuKConfiguration: WeightSourcing {
    static let modelFiles = ["config.json", "dit.safetensors", "vae.safetensors"]
    static let encoderFiles = ["config.json", "encoder.safetensors", "tokenizer.json", "tokenizer_config.json",
                               "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
                               "chat_template.json", "preprocessor_config.json"]

    public var weightSources: [WeightSource] {
        [WeightSource(role: "model", repo: modelRepo, revision: revision, matching: Self.modelFiles),
         WeightSource(role: "encoder", repo: encoderRepo, revision: encoderRevision, matching: Self.encoderFiles)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        let store = ModelStore(root: storeRoot)
        func present(_ explicit: URL?, _ repo: String, _ probe: String) -> Bool {
            if let explicit, fm.fileExists(atPath: explicit.appending(path: probe).path) { return true }
            if let dir = store.directory(for: repo), fm.fileExists(atPath: dir.appending(path: probe).path) { return true }
            return false
        }
        var missing = [WeightSource]()
        if !present(modelDirectory, modelRepo, "dit.safetensors") { missing.append(weightSources[0]) }
        if !present(encoderDirectory, encoderRepo, "encoder.safetensors") { missing.append(weightSources[1]) }
        return missing
    }

    /// Nil directories resolved to the store layout — what `load()` uses AFTER materialization.
    public func resolved(storeRoot: URL?) -> AuKConfiguration {
        var cfg = self
        let store = ModelStore(root: storeRoot)
        if cfg.modelDirectory == nil { cfg.modelDirectory = store.directory(for: modelRepo) }
        if cfg.encoderDirectory == nil { cfg.encoderDirectory = store.directory(for: encoderRepo) }
        return cfg
    }
}

extension AuKConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        let r = resolved(storeRoot: modelsRootDirectory)
        return [r.encoderDirectory?.appending(path: "encoder.safetensors"), r.modelDirectory?.appending(path: "dit.safetensors"),
                r.modelDirectory?.appending(path: "vae.safetensors")].compactMap { $0 }
    }
}
