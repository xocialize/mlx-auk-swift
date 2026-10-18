# AuK / AuK-Flash: MLX-Swift port spec

Decision **AB-D-0153**, task **AB-T-0325**. Reopens ENHANCEMENTS **E18** (evaluated 2026-09-12, PASS on the encoder licence). The operator reopened it on
2026-10-18 for the same engine posture as Qwen-Image 2.1 (AB-D-0085) and FLUX 3 Klein (AB-D-0151): a restricted,
named-only engine model that is not a product asset. This is a full port from Tencent's PyTorch release and Qwen's
original weights. No third-party MLX conversion is used, including vanch007/AuK-*-MLX, smcleod/AuK-MLX-8bit and
audio-cpp GGUF, because other authors' recent ports have had problems.

Upstream is pinned at `Tencent-Hunyuan/AuK@9481474` (2026-10-14). The weights are `tencent/AuK@790742b7`,
`tencent/AuK-Flash@575b92f0` and `Qwen/Qwen2.5-Omni-3B@f75b40e3`. The scratch side of the port (the upstream clone,
the fp32 goldens and the venvs) lives in `mlxengine-audio/WIP/auk-port/`. Everything needed to rebuild it is in
`Tools/`:
- `oracle-capture/dump_goldens.py --assets <clone>/assets/demo-input-audio --out <goldens>` regenerates the goldens
  bit for bit in about 30 minutes;
- `convert/convert.py --out <weights>` writes the bundles.

## 0. Status (2026-10-18): V0–V4 done, v0.1.0 published

| rung | outcome |
|---|---|
| V0 oracle | fp32 CPU goldens for 6 tasks × Base/Flash; outputs ASR-checked |
| V1 converter | bit-exact against upstream's folded weights (1,249 tensors, max \|Δ\| 0) |
| V2 Swift core | fp32 parity on 12/12 lanes: generated latent relL2 ≤ 6.6e-4, waveform ≤ 7.5e-3 |
| V2b tiers | bf16 / 8-bit / 4-bit encoder (and 8-bit DiT) all pass the pre-registered gate (§6–7) |
| V3 engine | `MLXAuK` on `tts` + `speechEdit`; 16/16 conformance; footprints from phys_footprint through MLXServeEngine |
| V4 publish | xocialize/mlx-auk-swift; mlx-community `AuK-bf16`, `AuK-Flash-bf16`, `Qwen2.5-Omni-3B-Thinker-{bf16,8bit,4bit}` plus a collection; hub smoke passed |

Numbers are in `MEASUREMENTS.md`; raw logs are in `receipts/`. Records: AB-D-0153, AB-T-0325, receipt AB-R-0586,
lesson AB-L-0297.

**Open.**
- AuK's content / speed / pitch / volume / timbre / de-accent edits need new `speechEdit` operations: ask AB-A-0188 to
  mlx-engine.
- Upstream's promised permissive encoder ([AuK#13](https://github.com/Tencent-Hunyuan/AuK/issues/13), team reply:
  *"plans to replace the current encoder with an alternative encoder under a permissive license"*). When it ships, port
  the new encoder and the DiT trained against it behind `AukConditionEncoder`; the package then becomes MIT.

Before the port (2026-09-12): E18 evaluated the model and passed on the encoder licence (`Docs/ENHANCEMENTS.md` E18).
No MLX code, conversion, or Python-MLX `qwen2_5_omni` existed anywhere. Third-party MLX copies (vanch007, smcleod,
audio-cpp GGUF) were deliberately not used.

## 1. Licence posture (decided by the operator; follows AB-D-0085 and AB-D-0151)

- DiT and VAE weights and code: **MIT**.
- The required encoder, Qwen2.5-Omni-3B, is under the **Qwen RESEARCH LICENSE**: non-commercial; §3 allows
  redistribution with the agreement and the NOTICE; §4b requires "Built with Qwen".
- C7: `weightLicense = LicenseRef-Qwen-Research`, the most restrictive component. It is declared in the package and
  never added to an allowlist. The CatalogEntry is `.restricted` and `.wip`, `named_only` behind
  `--allow-research` / `allowRestricted`, with `commercial_use: false` and an empty `preferredFor`. There is no
  MARKETING-HIGHLIGHTS entry.
- **Encoder seam.** The semantic encoder sits behind a protocol, `AukConditionEncoder`, which takes ChatML plus
  optional audio and returns fused hidden states `[B, nt, d_llm]` and a mask. The licence is reported per encoder.
  When Tencent ships the permissive encoder, a new conformer plus new DiT weights turn the package MIT without
  touching the engine wiring.
- Publishing is covered by the standing HF authorization:
  - DiT + VAE go to mlx-community under MIT.
  - The thinker (LLM + audio tower; no vision, talker or token2wav) goes to its own mlx-community repo. That repo
    carries the Qwen Research LICENSE, the NOTICE, a statement that the weights are modified, "Built with Qwen" and
    the HF licence tag `other`.

## 2. Architecture as read (verified against `config.yaml`)

**Pipeline** (`infer_auk.AukInfer.generate`):
1. Audio is converted to mono and resampled to 24 kHz with `torchaudio` Resample (sinc, Hann), so AB-L-0209 applies.
2. The VAE encodes the reference: `mean + randn·exp(log_std)`, then `(z − global_mean)/√global_log_std`. This step is
   **stochastic**, so the gates use the mean plus injected noise.
3. The Qwen2.5-Omni processor builds ChatML. The chat template runs with `add_generation_prompt`. Audio goes through
   `qwen_omni_utils.process_mm_info` to Whisper 128-mel features at 16 kHz. A text-only call appends
   `|<no_prompt_audio>|`.
4. The thinker forward pass produces `output_hidden_states`. The fusion is
   `Σ softmax(layer_weights[36])_i · LN(h_i) · layer_scale`. Each `LN` is a non-affine `F.layer_norm` with eps 1e-5,
   and the sum covers `hidden_states[1:]`. The thinker runs in bf16 upstream.
   - **Gotcha:** HF's last hidden state is *post-final-norm*. Confirm this on the pinned transformers 4.57.
5. CFM Euler (`torchdiffeq odeint`, method euler):
   - Base: `t = linspace(0,1,33)` plus sway −1.0. CFG 2.0 runs as a doubled batch:
     `v_c + (v_c − v_u)·2`. The uncond branch zeros the *projected* text and the ref latent and keeps `c_mask`.
   - Flash: `t_grid = [0, .0761205, .2928932, .6173166, 1]`, CFG off, sway None.
   - Noise is `torch.randn(dur, 64)` after `manual_seed(seed)`, so parity injects the noise.
6. The DiT predicts velocity on `[ref ; x_t]`. The output is sliced to the target, de-normalised, and passed to
   `inference_from_latents` (BigVGAN, clamp ±1). The output RMS is not restored, because `ref_rms` is unused.

**DiT `Flux2Edit`.** It is F5-TTS-lineage MMDiT code, not the diffusers FLUX block, so `flux2-klein-swift` is a donor
for structure only.
- dim 1536, 24 heads × 64, `ff_mult 2` gives a SwiGLU inner of 3072 (`linear_in` 1536→6144 without bias,
  `linear_out` without bias).
- 10 `MMDiTBlock` + 20 `DiTBlock`, `attn_mask_enabled: True`.
- `to_qkv` is fused and split with `.chunk(3)` (stacked, not interleaved). It has bias. QK-RMSNorm is per head.
  Joint attention concatenates **audio first, then text**.
- RoPE is x_transformers `RotaryEmbedding(64)`: rotate-half, positions from 0 **per stream** in the dual blocks, then
  0…(nt+n) over `[text ; audio]` in the single blocks.
- AdaLN-zero: SiLU→Linear(6d), non-affine LN with eps 1e-6, `x·(1+scale)+shift`. `AdaLayerNorm_Final` uses the
  order (scale, shift).
- Time embedding: F5 sinusoid with `scale=1000`, 256-dim, then Linear→SiLU→Linear.
- Text projection: `txt_proj` 2048→1536, then RMSNorm (affine). It is cached across steps.
- `AudioPromptEmbedding`:
  - Linear 64→1536, plus a residual `ConvPositionEmbedding` (2 × grouped Conv1d k31 g16 + Mish, masked between
    layers).
  - It runs separately on ref and x, then concatenates them.
- Attention outputs are zeroed by `audio_mask` / `c_mask` after the projection.

**VAE `BigVGANFlowVAE`**, 24 kHz, hop 480, 64-dim latent:
- Encoder:
  - It is non-causal. `Conv1d_S` is weight_norm with symmetric padding. Strides are 2,2,2,3,4,5 over channels
    12→768, followed by `ResStack(k3, base 2, 6)`.
  - **Gotcha:** the ResStack's `nn.LeakyReLU()` uses the *default slope 0.01*, while the encoder trunk uses 0.2.
  - The output conv produces 128 channels, split into μ and log σ.
- Decoder:
  - It is causal. `conv_pre` k7 is non-causal.
  - The causal ConvTranspose (k = 2·stride) trims `stride` from the end. Upsampling runs 1536→24 channels.
  - Each stage averages 3 `AMPBlock1` (kernels 3/7/11, dilations 1/3/5).
  - SnakeBeta (log-scale α, β) sits inside `Activation1d`. Its **up**-sampler is *non-causal* (replicate pad 5, trim
    15/15). Its **down**-sampler is *causal* (left pad 11). The filters are kaiser-sinc with 12 taps.
  - `conv_post` has no bias. The output is clamped to ±1.
- The coupling `flow` is training-only. It is dropped at conversion.
- weight_norm `(g, v)` is folded at conversion.
- The upstream `Conv1d` auto-transposes when `x.size(1) != in_channels`. That is a trap only for odd shapes; port the
  standard path.

**Thinker (Qwen2.5-Omni-3B).** Read from transformers 4.57.1 and confirmed in the V0 goldens.
- **M-RoPE collapses to 1-D RoPE here.** `get_rope_index` takes its 3-D path only when an image or video grid is
  passed. With audio and text alone it is `cumsum(attention_mask) − 1`, the same for all three sections. Audio tokens
  are plain sequential positions, so the LLM is a standard Qwen2: 36 × 2048, GQA 16/2, SwiGLU 11008, θ 1e6, RMS eps
  1e-6, untied embeddings (the LM head is unused).
- **`hidden_states[36]` is post-final-norm**, `self.norm` applied (`modeling_qwen2_5_omni.py:1643–1647`). The fusion
  uses layers 1…36, so the port must include the final RMSNorm on the last one. The goldens agree: mean |h| is 2.78
  at layer 35 and 2.02 at layer 36.
- **Audio tower** (`Qwen2_5OmniAudioEncoder`):
  - The mel is split into **chunks of 200 frames** (`n_window·2`); the tail chunk is shorter.
  - Each chunk is zero-padded and run through conv1 (k3 p1) + GELU, masked, then conv2 (k3 s2 p1) + GELU.
  - The sinusoid position embedding **restarts at 0 for each chunk**.
  - Attention is **block-diagonal per chunk** (`cu_seqlens`).
  - 32 pre-LN layers (k_proj without bias), exact-erf GELU, FFN 5120.
  - Then, over the whole clip: AvgPool1d(2), `ln_post`, `proj` 1280→2048. That gives 25 tokens/s (158 tokens for
    6.32 s).
  - The embeddings are `masked_scatter`ed into the `<|AUDIO|>` (151646) positions.
  - `audio_bos_eos_token` is unused on this path, because the bos/eos are text tokens 151647/151648.
- **Features:**
  - `WhisperFeatureExtractor`, 128 mel at 16 kHz, padded to **30,000 frames**. Only `feature_attention_mask.sum()`
    frames reach the tower.
  - The waveform comes from `librosa.load(sr=16000)`, which resamples with **soxr_hq**. That is a second resampler,
    separate from the VAE's torchaudio sinc path. Gate it on `input_features` with a tolerance, and save `audio16k`
    so the mel can be gated in isolation.
- **ChatML** (Qwen2.5-Omni template): the system turn defaults to "You are a helpful assistant.". The user turn is
  `instruction` followed by `<|audio_bos|><|AUDIO|><|audio_eos|>`, and the prompt ends with
  `<|im_start|>assistant\n` (add_generation_prompt). The processor expands `<|AUDIO|>` to N tokens.

## 3. Rungs

The plan as written before the work. Where the shipped result differs, the outcome line under the table says how.

| rung | what | gate |
|---|---|---|
| **V0 oracle** | venv with torch 2.7 / transformers 4.57 (as validated upstream). Run upstream inference on CPU in fp32 and dump goldens (§4) for 6 tasks: zero-shot TTS, instruct TTS, content edit, speed edit, emotion edit, denoise. Run each on Base and Flash with fixed injected noise. | runs and sounds right |
| **V1 converter** | torch→MLX safetensors: fold weight_norm, transpose convs, drop `flow.*` and `visual`/`talker`/`token2wav`, split `dit` / `vae` / `encoder`. Materialise before save. | key coverage `.noUnusedKeys`, round-trip tensor diff 0 |
| **V2 Swift core** `mlx-auk-swift` | `AukDiT`, `AukVAE` (enc + dec), `QwenOmniThinkerEncoder` (Qwen2 from mlx-swift-lm conventions plus a new audio tower), sampler with Base and Flash schedules, in-package resampler (sinc_interp_hann, AB-L-0209), Whisper mel. | per-module goldens (§4); end-to-end matched to the oracle |
| **V2b tiers** | bf16 DiT against the fp32 oracle; thinker bf16 / int8 / int4. **Named risk:** the 36-layer fused condition is not argmax-masked, so measure the drift of the fused hidden state and the end-to-end SIM/WER per tier. DiT int8. | pre-registered tolerance (set before measuring) |
| **V3 engine package** | `ModelPackage` on `Capability.speechEdit` (with TTS and instruct-TTS as modes; the mlx-engine ask if a mode is missing), two-layer licence gate (§1), `QuantFootprint.residentBytes` from `phys_footprint` (AB-L-0113), C0–C13 conformance. | conformance green |
| **V4 publish** | xocialize/mlx-auk-swift (MIT code, THIRD_PARTY_NOTICES). mlx-community `AuK-bf16` / `AuK-8bit`, `AuK-Flash-bf16` / `-8bit` (DiT + VAE), and `Qwen2.5-Omni-3B-Thinker-bf16` / `-8bit` (encoder, Qwen licence bundle). Group them in a Collection. | load-from-hub smoke |

The MES Audio Engine does not exist yet, so V3 targets MLXEngine `Capability.speechEdit` next to Step-Audio-EditX.
That is what a future audio engine would wrap, the same way the Image Engine wraps the image packages.

**As shipped.**
- **V3** serves two surfaces, `tts` (clone + voice design via `textDescription`) and `speechEdit`, in two packages,
  `auk` and `auk-flash`.
- **V4** publishes only bf16 DiT + VAE bundles. The quantised tiers are on the encoder (8- and 4-bit), because V2b
  showed DiT quantisation buys little (D vs B: −1.4 GB). So there is no `AuK-8bit`, and a 4-bit encoder was added.

## 4. Parity gates (fp32 CPU oracle; thresholds per the mlx-porting skill)

1. Thinker: token ids identical; audio features max-abs < 1e-4; per-layer hidden states; **fused condition** cos >
   0.9999 at fp32. Measure on the CPU stream first, given the E16 bf16 near-tie lesson.
2. DiT: one velocity on fixed inputs (text embeds, ref latent, x_t, t). Check per block, then the full pass < 1e-2
   (fp32 target < 1e-3).
3. VAE: encoder mean and log σ on a fixed clip; decoder on fixed latents. Audio max-abs and SNR.
4. Sampler: final latent with injected y0, for Base (64 evals) and Flash (4).
5. End to end: waveform SNR against the oracle with the same noise, plus WER (ASR) and speaker SIM per task.

## 5. Budget (from E18, an estimate, not a measurement)

| | resident estimate |
|---|---|
| bf16 total | ≈ 12 GB |
| 8-bit thinker | ≈ 8 GB |
| 4-bit thinker | ≈ 6.5 GB |

The speed estimate is RTF ≈ 1 for Base and ≈ 0.1 for Flash on an M5 Max. Declared footprints come only from
measurement.

**Measured** (phys_footprint through the engine): bf16 12.75 GB, 8-bit 9.46 GB, 4-bit 7.63–7.90 GB resident;
activation ≤ 1.03 GB; RTF Base ≈ 0.9, Flash ≈ 0.12. The estimate held for bf16 and was 1.1–1.5 GB low for the
quantised tiers: the embeddings and the audio-tower convs stay bf16.

## 6. V2b protocol (pre-registered 2026-10-18, before any tier was measured)

**Tiers.** All run on the GPU with MLX defaults (TF32 on), through the production path: the pipeline's own noise,
its own 24 kHz / 16 kHz sinc resampling and its own mel features.

| tier | DiT + VAE | encoder (thinker) |
|---|---|---|
| R | fp32, `MLX_ENABLE_TF32=0` (the reference; equal to the oracle, §7) | fp32 |
| A | bf16 | bf16 |
| B | bf16 | 8-bit (group size 64) |
| C | bf16 | 4-bit (group size 64) |
| D | 8-bit (block Linears only) | 8-bit |

**Matrix.** 6 tasks × {Base, Flash} × seeds {0, 1, 2} per tier, the same requests as the goldens.

**Metrics.**
- **Error rate:** WER for English, CER for Mandarin. mlx-whisper large-v3-turbo at temperature 0, with the fleet's
  normalisers (`speech-edit-eval/harness/score.py`). The reference text is the oracle's own transcript of each task.
- **SIM to source:** CAM++ (funasr campplus_cn_common) between the output and the source / reference clip, for the
  tasks that have one.
- **DNSMOS OVRL.**
- **Diagnostic only, not gated:** fused-condition drift (cos, relL2) of each encoder tier against the fp32 golden
  on the 12 goldens.

**Pass rule.** A tier passes against R when all three hold:
1. Pooled mean error rate is within **+2.0 points** absolute, and no (task, variant) mean is more than **+10
   points** worse.
2. Pooled mean SIM to source is within **−0.03**.
3. Pooled mean DNSMOS OVRL is within **−0.15**.

## 7. Results

V2 parity and V2b tiers are in `MEASUREMENTS.md`. In short:
- Tier R (fp32, TF32 off) equals the oracle, since the end-to-end matrix holds at fp32.
- Against R, every tier passes §6:
  - A bf16: Δerr −1.2 points, ΔSIM −0.023;
  - B 8-bit encoder: −1.7, −0.026;
  - C 4-bit encoder: −2.1, −0.022 (worst cell +1.5);
  - D 8-bit DiT + encoder: −1.4, −0.025.
- DNSMOS is within 0.03 in every tier.
- The named risk (quantised-thinker drift through the 36-layer fusion) did not materialise: the 4-bit thinker costs
  no more SIM than bf16.
- Scores: `receipts/v2b-scores.csv`.
