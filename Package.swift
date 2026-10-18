// swift-tools-version: 6.2
// mlx-auk-swift — Tencent Hunyuan's AuK / AuK-Flash (unified speech generation and editing) ported to Swift-MLX.
// One natural-language instruction covers zero-shot and instruct TTS, content edits (replace / insert / delete words),
// acoustic edits (pitch, speed, volume), paralinguistic edits (emotion, timbre, de-accent, nonverbal, whisper), and
// enhancement / separation. Port plan, licence posture and gates: mlxengine-audio/WIP/auk-port/PORTING-SPEC.md;
// decision AB-D-0153, task AB-T-0325.
//
// Licence: this package's code is MIT, and AuK's DiT and VAE weights are MIT. The REQUIRED semantic encoder
// (Qwen2.5-Omni-3B thinker) is under the Qwen RESEARCH LICENSE (non-commercial), so a pipeline that runs is
// LicenseRef-Qwen-Research. The encoder sits behind `AukConditionEncoder` for the permissive encoder upstream has
// announced.
//
// Pipeline (AuKCore, 1:1 with Tencent-Hunyuan/AuK@9481474):
//   • Encoder  — Qwen2.5-Omni thinker: Qwen2 36 × 2048 + chunked Whisper-shaped audio tower; ELMo fusion of all 36
//                hidden states (`CFMEdit.encodeText`).
//   • DiT      — Flux2Edit: 10 MMDiT + 20 single-stream blocks, dim 1536, conv-pos audio embedding, ref prepended.
//   • Sampler  — Euler flow: Base 32 steps × CFG 2 + sway −1; Flash 4 fixed steps, no CFG.
//   • VAE      — BigVGAN-Flow: 24 kHz, hop 480, 64-dim latents; causal anti-aliased SnakeBeta decoder.
//   • auk-gates — parity gates against the fp32 torch oracle (WIP/auk-port/oracle) + a render lane.
import PackageDescription

let package = Package(
    name: "mlx-auk-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "AuKCore", targets: ["AuKCore"]),
        .executable(name: "auk-gates", targets: ["auk-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "AuKCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/AuKCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "auk-gates",
            dependencies: [
                "AuKCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/auk-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
