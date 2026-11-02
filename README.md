# mlx-auk-swift

[AuK and AuK-Flash](https://huggingface.co/tencent/AuK) (Tencent Hunyuan, [arXiv:2609.08936](https://arxiv.org/abs/2609.08936)),
unified speech generation and editing, ported to Swift-MLX for Apple Silicon. One natural-language instruction covers:
- zero-shot and instruct TTS;
- content edits (replace / insert / delete words);
- acoustic edits (pitch, speed, volume);
- paralinguistic edits (emotion, timbre, de-accent, nonverbal, whisper);
- enhancement and separation.

> **⚠️ Research licence.** This package's code is MIT, and AuK's DiT and VAE weights are MIT. AuK *requires* the
> Qwen2.5-Omni-3B thinker as its semantic encoder, which is under the **Qwen RESEARCH LICENSE AGREEMENT**
> (non-commercial). So any working use of this package is research or evaluation only. The engine package declares
> `LicenseRef-Qwen-Research` and no permissive allowlist admits it. See `THIRD_PARTY_NOTICES.md`.

## Products

| product | what |
|---|---|
| `AuKCore` | The model: Flux2Edit DiT, BigVGAN-Flow VAE, Qwen2.5-Omni thinker encoder with the 36-layer fusion, the Euler flow sampler, and typed cookbook instructions (`AukInstruction`). `AukPipeline` runs one request. |
| `MLXAuK` | MLXEngine packages `AuKPackage` (`auk`) and `AuKFlashPackage` (`auk-flash`) on `tts` and `speechEdit` (contract 1.59.0: content, speed / pitch / volume, emotion, timbre, de-accent, whisper, denoise, dereverb and nonverbal edits; no transcript needed). |
| `auk-gates` | Parity gates against the fp32 torch oracle, the precision-tier matrix, engine validation, and the encoder quantiser. |

## Weights

| repo | contents | licence |
|---|---|---|
| [mlx-community/AuK-bf16](https://huggingface.co/mlx-community/AuK-bf16) | Base: DiT, VAE, config | MIT |
| [mlx-community/AuK-Flash-bf16](https://huggingface.co/mlx-community/AuK-Flash-bf16) | Flash: DiT, VAE, config | MIT |
| [mlx-community/Qwen2.5-Omni-3B-Thinker-bf16](https://huggingface.co/mlx-community/Qwen2.5-Omni-3B-Thinker-bf16) / [-8bit](https://huggingface.co/mlx-community/Qwen2.5-Omni-3B-Thinker-8bit) / [-4bit](https://huggingface.co/mlx-community/Qwen2.5-Omni-3B-Thinker-4bit) | the encoder | Qwen Research |

## Measured (M5 Max, mlx-swift 0.32.3)

Phys footprint through MLXServeEngine, plus speed:

| tier | resident | Base RTF | Flash RTF |
|---|---|---|---|
| bf16 | 12.8 GB | 0.89 | 0.13 |
| int8 encoder | 9.5 GB | 0.84 | 0.13 |
| int4 encoder | 7.6–7.9 GB | 0.83 | 0.12 |

Activation is ≤ 1.0 GB for every tier, measured after each request (a 5 ms sampler sees 1.9–2.5 GB inside one).

**Staged residency** (`AuKConfiguration(residency: .staged)`): the encoder and the DiT are read for each request and
freed after their stage, so only one large model is in memory at a time. The output is bit-identical. Peak phys drops
to 10.8 GB (bf16), 7.7 GB (int8) and 6.4–7.3 GB (int4), against 14.9 / 11.6–11.8 / 9.8–10.0 GB resident. Idle drops to
2.3–2.8 GB. Each request takes +0.2–0.45 s with a warm page cache, or about +1 s cold on a 13 GB/s disk. The engine
still charges the resident footprints until AB-A-0246 settles the declaration. Parity at fp32 against the upstream PyTorch code: generated latent relL2
≤ 6.6e-4 on 12 lanes. All three tiers pass a pre-registered quality gate (WER/CER, CAM++ SIM, DNSMOS). Details are
in `PORTING-SPEC.md` and `MEASUREMENTS.md`.

## Use

```swift
import AuKCore

let pipe = try await AukPipeline.load(modelDir: aukFlashDir, encoderDir: thinkerDir, dtype: .bfloat16)
let (samples, rate) = try WAV.read(take)
let edited = try pipe.generate(AukRequest(instruction: AukInstruction.replace(original: "but accepting what we cannot have",
                                                                             new: "and living well with dreams unmet").text(),
                                          audio: samples, sampleRate: rate, genSeconds: 7.0, seed: 1))
try WAV.write(edited, sampleRate: 24000, to: out)
```

Through MLXEngine, register `AuKFlashPackage.registration` (or `AuKPackage`) with an engine whose licence policy
admits research weights; the default `.permissiveOnly` flags it.

## Build

Build with the Xcode build system so MLX's metallib ships:
`swift build -c release --build-system swiftbuild` and `swift test --build-system swiftbuild`.
`Tools/` holds the Python oracle, the converter, the scorers, the run matrices and the publish staging (see `Tools/README.md`); `receipts/` holds the raw outputs behind `MEASUREMENTS.md`.
