---
library_name: mlx
license: mit
pipeline_tag: text-to-speech
base_model: tencent/AuK-Flash
tags:
  - mlx
  - mlx-swift
  - speech-editing
  - text-to-speech
---

# mlx-community/AuK-Flash-bf16

[tencent/AuK-Flash](https://huggingface.co/tencent/AuK-Flash) (Tencent Hunyuan; technical report
[arXiv:2609.08936](https://arxiv.org/abs/2609.08936)) converted to MLX for
[xocialize/mlx-auk-swift](https://github.com/xocialize/mlx-auk-swift).

AuK is a unified speech generation and editing model. One natural-language instruction covers:
- zero-shot and instruct TTS;
- content edits (replace / insert / delete words);
- acoustic edits (pitch, speed, volume);
- paralinguistic edits (emotion, timbre, de-accent, nonverbal, whisper);
- enhancement and separation.

Flash: the distilled 4-step recipe, CFG off (4 DiT forwards). RTF ≈ 0.13 bf16 on an M5 Max.

## ⚠️ Licence: read before use

The DiT and VAE in this repo are **MIT** (Tencent's `LICENSE`, included). AuK **requires** a semantic encoder, the
Qwen2.5-Omni-3B thinker, and that encoder is under the **Qwen RESEARCH LICENSE AGREEMENT**: research and evaluation
only, non-commercial. A pipeline that runs is therefore research-only, whatever this repo's own badge says. The encoder
is published separately with its licence:
[mlx-community/Qwen2.5-Omni-3B-Thinker-bf16](https://huggingface.co/mlx-community/Qwen2.5-Omni-3B-Thinker-bf16)
(also `-8bit` and `-4bit`). Tencent has said it plans a permissive encoder
([Tencent-Hunyuan/AuK#13](https://github.com/Tencent-Hunyuan/AuK/issues/13)).

## Files

| file | contents |
|---|---|
| `dit.safetensors` | Flux2Edit DiT (10 MMDiT + 20 single-stream blocks, dim 1536) plus the 36-layer fusion weights. bf16; the fusion logits, scale and RoPE frequencies are fp32. |
| `vae.safetensors` | BigVGAN-Flow VAE (24 kHz, hop 480, 64-dim latents). bf16 with fp32 normalisation statistics and anti-alias taps. Weight-norm folded; the training-only coupling flow removed. |
| `config.json` | Architecture, VAE and sampling configuration. |

Conv weights are in MLX (O, K, I) layout. Key names are upstream's verbatim. Converted from
`tencent/AuK-Flash@575b92f0` with `Tools/convert/convert.py`. A V1 gate checks it:
the fp32 conversion equals upstream's folded weights bit for bit.

## Parity

The reference is the upstream PyTorch code (`Tencent-Hunyuan/AuK@9481474`) run in fp32. Measured on six cookbook
tasks with the same noise, on Base and Flash:
- the Swift port's generated latent is within relL2 ≤ 6.6e-4 at fp32, and the waveform within ≤ 7.5e-3;
- bf16 against fp32 over 36 renders: error rate −1.2 points, CAM++ SIM −0.023, DNSMOS −0.01;
- the 8-bit and 4-bit encoders pass the same gate.

Receipts: [PORTING-SPEC and MEASUREMENTS](https://github.com/xocialize/mlx-auk-swift).

## Use (Swift)

```swift
import AuKCore
let pipe = try await AukPipeline.load(modelDir: aukDir, encoderDir: thinkerDir, dtype: .bfloat16)
let wav = try pipe.generate(AukRequest(instruction: AukInstruction.emotion("happy").text(),
                                       audio: samples, sampleRate: 24000, seed: 1))
```

This repo contains modified (converted) weights; Tencent does not endorse this conversion.
