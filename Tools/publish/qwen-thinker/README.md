---
library_name: mlx
license: other
license_name: qwen-research
license_link: LICENSE
base_model: Qwen/Qwen2.5-Omni-3B
tags:
  - mlx
  - mlx-swift
  - qwen2.5-omni
---

# mlx-community/Qwen2.5-Omni-3B-Thinker-__TIER__

The **thinker** of [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B): the Qwen2 language model
(36 × 2048) and the Whisper-shaped audio tower. It is converted to MLX as the semantic encoder of
[AuK](https://huggingface.co/mlx-community/AuK-bf16) in [xocialize/mlx-auk-swift](https://github.com/xocialize/mlx-auk-swift).
The vision tower, LM head, talker and token2wav are not included. This is an encoder for its hidden states, not a chat
model.

__QUANT__

## ⚠️ Licence

**Qwen RESEARCH LICENSE AGREEMENT**: research or evaluation use only, non-commercial. The full text is in
[LICENSE](LICENSE), and the attribution notice in [NOTICE](NOTICE). For commercial use, request a licence from Alibaba
Cloud.

These are **modified files**: converted (and, for -8bit / -4bit, quantised) from the original.

**Built with Qwen.**

## Files

- `encoder.safetensors` with `thinker.model.*` and `thinker.audio_tower.*` keys; audio conv weights in MLX (O, K, I)
  layout.
- `config.json` (text, audio and token ids, plus `quantization` when quantised).
- The original tokenizer, chat template and preprocessor files, unchanged.

## Parity (as AuK's encoder)

At fp32 against the transformers 4.57.1 reference:
- token ids identical;
- Whisper features relL2 1.5e-6;
- audio tower 3.9e-6;
- last hidden state 2.2e-5;
- AuK's fused 36-layer condition 3.8e-6.

As AuK's encoder, the bf16, 8-bit and 4-bit tiers all pass AuK's pre-registered quality gate (error rate, CAM++ SIM,
DNSMOS against fp32).
