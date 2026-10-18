// WeightIO.swift — safetensors in, key contract enforced (the mlx-step-audio-editx-swift idiom): load on the CPU stream
// (a multi-GB lazy load bound to the GPU stream can hold one command buffer past the Metal watchdog), cast floating
// tensors to the requested dtype, then compare the module's flattened parameter paths with the on-disk keys — 0 missing,
// 0 unused, or throw — before `update(verify: [.noUnusedKeys, .shapeMismatch])`. A partial load is the silent-failure
// class, so there is no tolerant path.
//
// The bundle layout is the V1 converter's (WIP/auk-port/convert/convert.py): upstream key names verbatim, conv weights
// already MLX (O, K, I), weight-norms folded, VAE `flow.*` and the thinker's vision / talker / lm_head dropped.

import Foundation
import MLX
import MLXNN

public enum AuKError: Error, CustomStringConvertible {
    case missingFile(String)
    case keyContract(component: String, missing: [String], unused: [String])
    case badConfig(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "missing file: \(p)"
        case .keyContract(let c, let m, let u):
            return "\(c): key contract violated — missing \(m.count) \(m.prefix(6)), unused \(u.count) \(u.prefix(6))"
        case .badConfig(let s): return "bad config: \(s)"
        case .invalidInput(let s): return "invalid input: \(s)"
        }
    }
}

public enum WeightIO {
    /// Loads a safetensors file on the CPU stream. Floating tensors are cast to `dtype` unless their key matches
    /// `keepFP32` (norm statistics, filter taps, fusion logits stay fp32 as the converter wrote them).
    public static func load(_ url: URL, dtype: DType?, keepFP32: (String) -> Bool = { _ in false }) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw AuKError.missingFile(url.path) }
        return try Device.withDefaultDevice(.cpu) { () throws -> [String: MLXArray] in
            var out = [String: MLXArray]()
            for (k, v) in try loadArrays(url: url) {
                let floating = [DType.float16, .bfloat16, .float32].contains(v.dtype)
                if floating, let d = dtype, !keepFP32(k), v.dtype != d {
                    out[k] = v.asType(d)
                } else {
                    out[k] = v
                }
            }
            eval(Array(out.values))
            return out
        }
    }

    /// Strips `prefix` from every key that has it and drops the rest.
    public static func scoped(_ arrays: [String: MLXArray], prefix: String) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (k, v) in arrays where k.hasPrefix(prefix) { out[String(k.dropFirst(prefix.count))] = v }
        return out
    }

    /// The key contract, then the update.
    public static func apply(_ arrays: [String: MLXArray], to module: Module, component: String) throws {
        let expected = Set(module.parameters().flattened().map(\.0))
        let onDisk = Set(arrays.keys)
        let missing = expected.subtracting(onDisk).sorted()
        let unused = onDisk.subtracting(expected).sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw AuKError.keyContract(component: component, missing: missing, unused: unused)
        }
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: [.noUnusedKeys, .shapeMismatch])
        module.train(false)
        eval(module.parameters())
    }
}

public enum ConfigIO {
    public static func load<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        guard FileManager.default.fileExists(atPath: url.path) else { throw AuKError.missingFile(url.path) }
        return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }
}
