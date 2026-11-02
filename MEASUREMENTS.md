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

## Contract 1.59.0 — the AuK edit kinds (2026-11-01)

mlx-engine-swift 0.67.0 → 0.74.0 (AB-A-0188, AB-D-0200). `MLXAuK` now declares fourteen speechEdit kinds:
- emotion, style, denoise, dereverberate;
- replaceWords, insertWords, removeWords;
- speed, pitch and volume, on AuK's trained grids;
- timbre, deAccent;
- addNonverbal and removeNonverbal, for breath, laugh, cough and humming.

It also declares `transcript: .unused` and `supportsTargetDuration: true`.

Without a `targetDuration`:
- a speed edit scales the source length by 1 / factor;
- a content edit uses upstream's byte ratio when the transcript is known, else the speaking-rate estimate of the
  changed words;
- every other edit keeps the source length.

`removeNonverbal(sound: nil)` is refused, because the cookbook's remove template names one sound. Offline conformance:
21 / 21.

**Live, `auk-gates validate`, bf16, M5 Max** (`receipts/validate-1.59.log`). The run adds four lanes on the cookbook's
own inputs, plus an off-grid refusal:

| lane | input → output | Flash RTF | Base RTF |
|---|---|---|---|
| replaceWords, targetDuration 7.0 | 7.01 → 7.00 s | 0.10 | 0.71 |
| speed ×1.5, no targetDuration | 11.00 → 7.34 s | 0.11 | 0.86 |
| pitch +2 | 5.50 → 5.50 s | 0.11 | 0.73 |
| removeNonverbal humming, targetDuration 22.0 | 23.88 → 22.00 s | 0.12 | 0.73 |

- ASR (mlx-whisper large-v3-turbo): the replace edit reads "…not fighting for what we want and living well with dreams
  unmet?" on both variants, and the speed edit keeps every word.
- An off-grid `.speed(1.1)` is refused by the engine before admission.
- Resident memory: 12.71 GB (Flash) and 12.78 GB (Base). Peak phys: 13.62 and 13.77 GB, inside the declared bf16
  footprint.
- Not measured here:
  - the int8 and int4 tiers on the new lanes;
  - the edit kinds that have no lane (insert, remove, volume, timbre, de-accent, dereverb, add nonverbal);
  - the pitch shift itself, in cents.

  The cookbook verifier in AB-T-0416 is for those.

## Stage profile — where a call's time goes (2026-11-01, AB-T-0415)

`auk-gates profile` runs the six golden tasks with `AukPipeline.generate`, on the production path: the pipeline's own
noise, resampling and features. Settings:
- GPU, TF32 at MLX's default, seed 0.
- One untimed warm-up per task, then 5 timed runs. Each figure is the lowest of the 5.
- Each config runs in a fresh process, holding the AuK GPU lock alone (`Tools/run_profile.sh`).
- Raw per-run log: `receipts/stage-profile.log`.

**How each stage is bounded.** `generate` already calls `eval` at the end of every stage, then `capture`. The profile
timestamps those `capture` calls, so each interval is finished GPU work, not lazy graph building:

| stage | from → to | what it holds |
|---|---|---|
| ref | call start → `ref_latents` | sinc resample to 24 and 16 kHz (CPU, ≈ 1 ms, timed separately), VAE encode |
| encoder | → `fused` | Whisper features, tokenizer, Qwen thinker (36 layers), ELMo fusion; `eval` after every layer |
| ODE | → `gen_latent` | the Euler loop; `eval(x)` after every step, which `onStep` timestamps |
| decode | → `audio_out` | denormalise + BigVGAN-Flow decoder |
| host | → return | the fp32 copy to `[Float]`, < 1 ms |

**Pooled over the six tasks** (32.9 s of audio in all). Each cell is the sum, over the tasks, of each task's lowest
time for that stage:

| config | ref | encoder | ODE | VAE decode | total | RTF | decode share: pooled (per-task range) |
|---|---|---|---|---|---|---|---|
| Flash bf16 | 0.07 s | 0.43 (12 %) | 1.94 (53 %) | 1.25 | 3.73 | 0.113 | **34 %** (23–39 %) |
| Flash 8-bit encoder | 0.07 | 0.44 (12 %) | 1.87 (52 %) | 1.23 | 3.60 | 0.110 | **34 %** (23–38 %) |
| Base bf16 | 0.07 | 0.50 (2 %) | 23.81 (93 %) | 1.22 | 25.60 | 0.779 | 5 % (4–6 %) |
| Base 8-bit encoder | 0.07 | 0.53 (2 %) | 23.66 (93 %) | 1.21 | 25.47 | 0.775 | 5 % (4–6 %) |

**Per task, Flash bf16** (lowest time per stage, in seconds; the bracket is that stage's share of the call):

| task | audio | ref | encoder | ODE (per step) | decode | total |
|---|---|---|---|---|---|---|
| content | 7.00 s | 0.012 | 0.063 | 0.324 (0.081) | 0.259 (39 %) | 0.659 |
| emotion | 6.32 | 0.011 | 0.069 | 0.303 (0.076) | 0.236 (38 %) | 0.618 |
| itts (no ref) | 1.70 | — | 0.042 | 0.178 (0.044) | 0.066 (23 %) | 0.288 |
| se | 4.98 | 0.010 | 0.072 | 0.298 (0.074) | 0.189 (33 %) | 0.571 |
| speed | 6.86 | 0.018 | 0.089 | 0.425 (0.106) | 0.266 (33 %) | 0.798 |
| zs | 6.00 | 0.017 | 0.090 | 0.414 (0.103) | 0.232 (31 %) | 0.791 |

**Reading.**
- **The VAE decoder does not dominate Flash here.** Pooled it is 34 % of the call; its worst task share is 39 %
  (content); the ODE is 52–53 %. AB-T-0415's 40 % threshold is not met, so no decoder-optimisation task is filed.
  Upstream's ~60 % decode (fp32 Python-MLX, M4 Pro, 4 s clip) does not reproduce on bf16 Swift on an M5 Max.
- **Decode cost is linear in output length**: 36–41 ms per second of audio in every config and task. Its share
  therefore depends on the ODE cost per step, which follows the whole DiT sequence (reference + target), not the
  output alone:
  - 0.044 s per step for 1.7 s of instruct TTS with no reference;
  - 0.074–0.106 s with a 5–11 s reference.
- Not measured here: calls near the 30 s ceiling. There the ODE's attention should grow faster than the linear
  decoder, so the decode share should fall further. A long-clip sweep would confirm it.
- **Base is ODE-bound**: 32 CFG steps take 92–94 % of the call. The doubled batch makes each step 1.1–1.7 × a Flash
  step.
- **The 8-bit encoder buys no speed.** Encoder time is the same as bf16 within noise (0.44 vs 0.43 s pooled for
  Flash). It is one short prefill, and its cost is mostly not weight reads. Tier B's value is memory (V2b: 9.4 vs
  12.6 GB), not latency.
- RTF from this pass (Flash 0.11, Base 0.78) sits at the low end of V2b / V3's Flash 0.11–0.14 and Base ≈ 0.9:
  those were single runs, these are the lowest of 5.
- Spread: within a task the slowest of 5 runs is at most 22 % over the fastest. The low end is reported.
- The first call in a process is dominated by kernel compile (Flash content warm-up: ref 1.5 s, encoder 2.4 s, total
  6.1 s). That is the cost `AuKConfiguration.warmUp` exists to pay at load.
