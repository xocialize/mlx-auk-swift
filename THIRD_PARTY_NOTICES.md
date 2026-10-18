# Third-party notices

This package translates code from, and loads weights derived from, the projects below. Their licences travel with it.

## AuK / AuK-Flash: Tencent-Hunyuan/AuK (code), tencent/AuK and tencent/AuK-Flash (weights)

MIT, Copyright (C) 2026 Tencent. The full text is in `LICENSES/MIT-tencent-auk.txt`.
- Translated upstream files: `model/flux2_edit.py`, `model/modules.py`, `model/cfm_edit.py`,
  `model/vae/bigvgan_flow_vae.py`, `model/vae/modules/{commons,bigvgan}/*`, and `infer/infer_auk.py`.
- The instruction templates in `AukInstruction.swift` are from `docs/COOKBOOK.md`.
- Upstream's own attributions in those files: BigVGAN (NVIDIA, MIT), HiFi-GAN (jik876, MIT), alias-free-torch
  (junjun3518, Apache-2.0), julius (adefossez, MIT), and x_transformers' rotary embedding (lucidrains, MIT).

## Qwen2.5-Omni-3B: Qwen/Qwen2.5-Omni-3B (the semantic encoder's weights)

**Qwen RESEARCH LICENSE AGREEMENT**: research and evaluation only, non-commercial. The full text is in
`LICENSES/Qwen-Research-License-Agreement.txt`.
- Notice: "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights
  Reserved."
- AuK cannot run without this encoder, so any use of this package is subject to that agreement. **Built with Qwen.**
- The thinker's architecture is translated from transformers' `models/qwen2_5_omni/modeling_qwen2_5_omni.py`
  (Apache-2.0, Hugging Face).

## mlx-step-audio-editx-swift (xocialize, MIT)

`Resample.swift`, `Signal.swift`, `NPY.swift`, `WAV.swift` and `AuKAudioIO.swift` are lifted from it; they are
parity-locked there.
