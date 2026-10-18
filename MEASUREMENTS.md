# AuK port — measurements

M5 Max, 128 GB, macOS 27.2, mlx-swift 0.32.3. Plan and gates: `PORTING-SPEC.md`. Decision AB-D-0153, task AB-T-0325.
Code: `mlxengine-audio/PROD/AuK/mlx-auk-swift`. Raw logs: `receipts/`.

## V0 — fp32 torch oracle (2026-10-18)

The oracle is upstream `Tencent-Hunyuan/AuK@9481474` with torch 2.7.1, transformers 4.57.1 and x_transformers
2.31.8. Everything runs in fp32 on the CPU; the thinker is upcast from upstream's bf16 load. Seeds: sampler 1234, VAE
reference noise 4321.
- Six cookbook tasks on Base and Flash: zero-shot TTS, instruct TTS, content replace, speed ×1.5, emotion→happy, and
  speech enhancement (zh, 8 kHz source).
- Wall time on CPU: Flash 4–20 s per task, Base 45–184 s.

**Content check** (mlx-whisper large-v3-turbo on the oracle outputs). Every output says what the task asked for.
- The content edit reads "…not fighting for what we want and living well with dreams unmet?".
- Instruct TTS reads "Welcome home. How was work today?".
- The zh enhancement keeps its words.
- Base speed dropped one word ("my fellow, ask not…"). That is model behaviour, not the oracle.

## V1 — converter (2026-10-18)

- `gate_v1.py`: the converted fp32 tensors equal upstream's own folded weights (`remove_weight_norm`) bit for bit:
  max |Δ| 0 over 829 VAE and 420 DiT tensors. Key sets are equal.
- The bf16 bundles differ only by rounding (VAE 7.4e-3, DiT 2.6e-3 max).
- Sizes:

| bundle | fp32 | bf16 |
|---|---|---|
| DiT | 6.12 GB | 3.06 GB |
| VAE | 0.59 GB | 0.29 GB |
| encoder (thinker LLM + audio tower) | 14.9 GB | 7.45 GB |

## V2 — Swift core parity (2026-10-18)

**Per module**, fp32, CPU stream, Flash emotion golden unless noted:

| gate | relL2 | cos |
|---|---|---|
| VAE encoder mean / log σ | 7.5e-6 / 1.6e-6 | 0.9999998 |
| VAE normalised latents (oracle noise) | 1.2e-5 | 0.9999998 |
| VAE decoder waveform | 5.4e-6 | 0.9999999 |
| DiT velocity (Flash, single branch) | 2.0e-6 | 1.0000000 |
| DiT velocity (Base content, CFG doubled batch) | 4.8e-7 | 0.9999999 |
| DiT velocity (Base instruct TTS, no reference) | 4.1e-7 | 0.9999999 |
| Whisper features (from the oracle's 16 kHz audio) | 1.5e-6 | 1.0000001 |
| audio tower | 3.9e-6 | 0.9999998 |
| token ids | identical (185 / 185) | — |
| hidden state 36 (post-norm) | 2.2e-5 | 1.0000000 |
| fused condition | 3.8e-6 | 1.0000000 |

**End to end** on the oracle's noise, GPU, fp32 with `MLX_ENABLE_TF32=0`, all 12 lanes:
- generated latent relL2 ≤ 6.6e-4 (Flash enhancement); 10 of 12 lanes ≤ 4.2e-4;
- waveform relL2 ≤ 1.2e-3, except Flash emotion at 7.5e-3, where the decoder amplifies a 1.4e-4 latent difference.

| lane | gen latent relL2 | audio relL2 | wall (s) / audio (s) |
|---|---|---|---|
| flash zs | 1.6e-4 | 2.4e-4 | 5.40 / 6.00 (first call, cold kernels) |
| flash itts | 6.9e-6 | 1.4e-5 | 0.49 / 1.70 |
| flash content | 9.7e-5 | 2.6e-4 | 1.30 / 7.00 |
| flash speed | 5.1e-5 | 1.2e-4 | 1.53 / 6.86 |
| flash emotion | 1.4e-4 | 7.5e-3 | 1.35 / 6.32 |
| flash se | 6.6e-4 | 1.7e-4 | 1.04 / 4.98 |
| base zs | 4.2e-4 | 1.2e-3 | 12.37 / 6.00 |
| base itts | 3.8e-5 | 5.8e-5 | 2.50 / 1.70 |
| base content | 3.0e-5 | 4.8e-5 | 10.26 / 7.00 |
| base speed | 2.6e-5 | 2.0e-4 | 12.72 / 6.86 |
| base emotion | 1.0e-4 | 3.1e-4 | 8.61 / 6.32 |
| base se | 1.1e-5 | 2.1e-5 | 7.39 / 4.98 |

**Bugs the gates caught:**
1. **Reference-latent length.** The VAE encoder emits one frame more than `len // 480` on 4 of 5 clips (351 vs 350
   for content). Upstream keeps that frame in the sequence and key-masks it (`lens_to_mask(ref_lens, length=Np)`).
   Cropping it shifted the target's RoPE positions by one: Flash content call-0 relL2 3e-2, waveform 0.13. The fix
   keeps the frame with the mask.
2. **Golden layout.** `randn_like` of a chunked view keeps the view's strides, and `np.save` then writes Fortran
   order. The goldens are now saved C-contiguous.

**GPU precision.** With MLX's default `MLX_ENABLE_TF32=1`, fp32 GPU matmuls cost about 1e-3 relL2 per GEMM
(AB-D-0114). That gives VAE encoder mean 7e-2, fused condition 3.9e-3, and Flash generated latent 0.17–0.51 against
the oracle. With TF32 off the GPU equals the CPU, so the drift is precision, not a port bug. The flow model turns
small condition differences into a different sample of the same content. The bf16 renders say the right words (ASR),
and V2b measures quality.

## V2b — precision tiers (2026-10-18; protocol pre-registered in PORTING-SPEC §6)

The production path (own noise, own sinc resampling, own mel), GPU with TF32 at MLX's default. Matrix: 6 tasks ×
{Base, Flash} × seeds {0, 1, 2} = 36 renders per tier.
- Scored with mlx-whisper large-v3-turbo against the oracle transcripts, CAM++ SIM to the source, and DNSMOS OVRL.
- Scores: `receipts/v2b-scores.csv`. Log: `receipts/v2b-score.log`.

| tier | error rate | SIM to source | DNSMOS | Base RTF | Flash RTF | MLX peak | verdict |
|---|---|---|---|---|---|---|---|
| R fp32 (TF32 off) | 5.32 % | 0.789 | 3.202 | 1.42 | 0.21 | 23.7 GB | reference |
| A bf16 | 4.09 % (−1.2) | 0.766 (−0.023) | 3.195 | 0.89 | 0.13 | 12.6 GB | PASS |
| B bf16 + 8-bit encoder | 3.59 % (−1.7) | 0.763 (−0.026) | 3.189 | 0.84 | 0.13 | 9.4 GB | PASS |
| C bf16 + 4-bit encoder | 3.23 % (−2.1; worst cell +1.5) | 0.767 (−0.022) | 3.174 | 0.83 | 0.12 | 7.7 GB | PASS |
| D 8-bit DiT + 8-bit encoder | 3.96 % (−1.4) | 0.764 (−0.025) | 3.178 | 0.77 | 0.11 | 8.0 GB | PASS |

**Reading.**
- No (task, variant) cell's error rate is more than 1.5 points worse than fp32. Most of the error comes from two
  cells, the same in every tier, R included:
  - Base emotion: ASR hears "social vacation", the source's own wording;
  - the zh enhancement clip.
- SIM drops 0.022–0.026 in every reduced-precision tier, and bf16 alone accounts for nearly all of it. The named V2b
  risk did not materialise: a 4-bit thinker costs no more SIM than a bf16 one.
- The 36-layer fused condition is robust to quantisation in practice.
- Shipped tiers: DiT and VAE bf16; encoder bf16 / 8-bit / 4-bit (A / B / C). D saves 1.4 GB over B at a small speed
  gain; not shipped.

## V3 — engine package (2026-10-18)

- `MLXAuK` provides `AuKPackage` (`auk`) and `AuKFlashPackage` (`auk-flash`) on `tts` and `speechEdit`.
- The licence is declared as `LicenseRef-Qwen-Research` (additional: MIT). It is not allowlisted;
  `LicensePolicy.permissiveOnly` refuses it, and the default advisory engine registers it with one advisory.
- Offline conformance: 16 / 16 tests (manifest, licence posture, MAT per tier × variant, CAN-1..3, request plane,
  cookbook templates).

**Live, `auk-gates validate`.** Run through `MLXServeEngine(policy: .any)`: register → prepare (warm-up incl.) → TTS
clone, TTS design, edit→happy, edit→whisper, edit→denoise → undeclared-label refusal → evict → second cycle. phys
footprint (`receipts/validate.log`):

| | resident | peak phys | activation | 2nd-cycle re-prepare | RTF (5 runs) |
|---|---|---|---|---|---|
| flash bf16 | 12.76 GB | 13.56 | 0.80 | = first | 0.11–0.14 |
| flash int8 | 9.49 | 10.46 | 0.96 | = first | ≈ 0.12 |
| flash int4 | 7.90 | 8.78 | 0.88 | = first | ≈ 0.12 |
| base bf16 | 12.75 | 13.79 | 1.03 | = first | ≈ 0.9 |
| base int8 | 9.46 | 10.39 | 0.92 | 9.63 vs 9.47 | ≈ 0.9 |
| base int4 | 7.63 | 8.62 | 0.98 | 7.69 vs 7.63 | 0.79–1.06 |

- A 27 s edit (the declared ceiling is 30 s) leaves the MLX peak where the 6 s edits put it: Base 12.60 GB at RTF
  0.82, Flash 12.49 GB at RTF 0.11.
- Declared footprints: bf16 12.9 GB, int8 9.6 GB, int4 7.8 (Base) / 8.0 (Flash) GB, activation 1.5 GB.
- **After evict.** phys stays at 2.5–9.6 GB with MLX active = cache = 0, and `malloc_zone_pressure_relief` frees
  nothing. The second cycle re-peaks at the first cycle's level, so this is process memory the next load reuses, not a
  leak.
- **Bug found:** the gate CLI's `DispatchSemaphore.wait()` on the main thread deadlocked the engine's inference actor
  (0 % CPU); fixed with top-level `await`.
