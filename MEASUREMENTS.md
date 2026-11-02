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
  **Superseded 2026-11-01:** these readings were taken after each request and miss the in-request peak. See
  "Staged residency": activation is now 2.8 GB and Base int4 is now 8.1 GB.
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

## Staged residency (2026-11-01, AB-T-0414)

`AuKConfiguration.residency = .staged` (default `.resident`). This is upstream's `--sequential`
(Tencent-Hunyuan/AuK `feat/mlx-apple-silicon` @ 6943a1e), reimplemented. Between requests, only the VAE (0.29 GB), the
two fusion tensors (`layer_weights`, `layer_scale`) and the tokenizer stay loaded. Each request:
1. computes the VAE reference latents, as before;
2. reads the Qwen2.5-Omni Thinker, conditions, then frees it and clears the MLX cache;
3. reads the DiT, samples, then frees it and clears the cache;
4. decodes with the VAE.

`AukStage` holds a model for the pipeline's lifetime or builds it for one `use`. `generate`'s signature is unchanged.
The raw outputs are in `receipts/staged-residency.txt`. The runner is `Tools/run_staged.sh bit|mem`. Measured on an M5
Max with mlx-swift 0.32.3, under the AuK GPU lock.

**Bit identity: PASS on every lane** (`auk-gates staged`):
- Lanes: 2 variants × 3 tiers (bf16, int8, int4 encoder bundles).
- Requests per lane: 6 V2 golden tasks × 2 paths.
  - The production path: own noise and resampling, seed 0.
  - The oracle-noise path: the e2e lane's injected y0, VAE noise and features. This covers the other conditioning
    branch.
- Every staged sample equals the resident sample bit for bit (max|Δ| 0): 96 staged renders against 72 resident.
- bf16 adds a cold-page-cache staged pass (24 of the 96), and it is also identical.

**Peak phys_footprint through the engine** (`auk-gates validate`, one process per run):
- Each run: MLXServeEngine register → prepare (warm-up included) → 9 requests → evict → a second cycle.
- "Peak" is the kernel's lifetime `ledger_phys_footprint_peak` for the process. A 5 ms sampler inside each request
  agrees with it to within 0.08 GB.
- "Idle" is phys after prepare, which is what stays charged between requests.
- Staged's idle is ~0.4 GB of MLX arrays plus ~2 GB of process memory left by the warm-up.

| | resident idle | resident peak | staged idle | staged peak | peak saved |
|---|---|---|---|---|---|
| flash bf16 | 12.76 GB | 14.89 | 2.35 | **10.75** | −4.14 |
| flash int8 | 9.45 | 11.61 | 2.37 | **7.66** | −3.95 |
| flash int4 | 7.91 | 9.83 | 2.28 | **6.39** | −3.44 |
| base bf16 | 12.51 | 14.92 | 2.72 | **10.79** | −4.13 |
| base int8 | 9.19 | 11.75 | 2.56 | **7.68** | −4.07 |
| base int4 | 7.97 | 10.04 | 2.82 | **7.34** | −2.70 |

- **Which stage sets the peak** (MLX peak per request, `auk-gates staged`):
  - bf16: the encoder stage, at 8.25 GB.
  - int8 and int4: the DiT stage, at 5.15 GB (Flash) and 5.27 GB (Base, with CFG's doubled batch).
  - Per-request staged peaks vary with input length: bf16 8.96–10.77 GB, int8 5.84–7.66, int4 5.26–7.32. The 22 s
    nonverbal edit is the largest.
- **On a 16 GB Mac:** int8 and int4 staged peak below 8 GB, and bf16 staged peaks at 10.8 GB. Upstream reports 8-bit
  9.1 → 6.1 GB, with their DiT also 8-bit; ours keeps the DiT bf16.
- **Finding on the resident declaration.** The 5 ms sampler shows the resident in-request peak is 1.9–2.5 GB over
  idle (2.56 GB at most, base int8). V3's after-request readings showed 0.80–1.03 GB. The declared 1.5 GB
  `peakActivationBytes` left the totals short by up to ~0.75 GB: base int4 measured 10.04 GB against 7.8 + 1.5 =
  9.3 GB, and base bf16 14.92 against 14.4. Base int4 idle also measured 7.97 GB against its declared 7.8.
  **Fixed:** activation is now 2.8 GB on every tier, and Base int4 resident is now 8.1 GB. Every measured peak sits
  ≥ 0.85 GB under resident + activation, and `testFootprintsAreTheMeasuredTiers` checks the peaks.

**Per-request latency added by the reloads** (`auk-gates staged`, 12 requests per pass, means):
- The load columns time the encoder and DiT safetensors reads, cast and key check.
- "Added" is staged wall minus resident wall for the same request. It includes the loads, plus ~0.1–0.15 s of fresh
  buffer allocation after each `clearCache`.

| | resident wall | staged warm: added (encoder + DiT load) | staged cold: added (encoder + DiT load) |
|---|---|---|---|
| flash bf16 | 0.59 s | +0.44 s (0.20 + 0.09) | +1.03 s (0.59 + 0.25) |
| flash int8 | 0.60 | +0.35 (0.16 + 0.10) | — |
| flash int4 | 0.59 | +0.30 (0.12 + 0.10) | — |
| base bf16 | 4.25 | +0.41 (0.19 + 0.09) | +0.97 (0.59 + 0.25) |
| base int8 | 4.26 | +0.33 (0.16 + 0.09) | — |
| base int4 | 4.34 | +0.20 (0.12 + 0.10) | — |

- **Through the engine** (the validate runs above), staged adds a mean +0.23 to +0.34 s per request on every tier, with
  the page cache warm.
- **On wall clock:** for Flash, staged makes each request ~1.5–1.8× longer. For Base, it adds 5–10 %.
- **Cold page cache.** Before every staged request, `encoder.safetensors` and `dit.safetensors` were evicted
  (msync `MS_INVALIDATE`), and mincore confirmed 0 % resident.
  - The eviction is real: after it, the file reads at 13.1 GB/s, the same as a file never read, against 26–27 GB/s
    cached (`coldprobe` in the receipt).
  - This volume is a 13 GB/s external PCIe SSD. Cold cost scales with read speed: a staged bf16 request reads 10.5 GB
    (int8 7.3, int4 5.6), so at 3 GB/s expect about +3.5 s.
- **Gate phys excluded.** The gate's own phys readings are not used. Its process has no MLX cache limit, so the
  resident pass climbs 17 → 37 GB of buffer cache while the MLX peak stays flat. The engine caps the pool, and the
  validate runs show no such growth.

**Declaration** (decided with mlx-engine in AB-A-0246, option (A), no contract change). `AuKConfiguration` adopts
`FootprintConfigured`. A `.resident` configuration returns no hints, so the engine charges the manifest's quant-keyed
footprint. A `.staged` one returns:

| | `residentBytesHint` (persistent) | `peakActivationBytesHint` (transient) | `expectedWeightReadBytesPerRunHint` |
|---|---|---|---|
| bf16 | 3.0 GB | 8.0 GB | 10.51 GB |
| int8 | 3.0 GB | 5.0 GB | 7.32 GB |
| int4 | 3.0 GB | 4.6 GB | 5.62 GB |

- **Floor:** the measured idle (2.28–2.82 GB) rounded up.
- **Transient:** the worst measured staged peak minus 3.0 GB, rounded up.
- **Read:** encoder.safetensors plus dit.safetensors.
- **At admission:** an idle staged AuK is charged 3.0 GB, so other models co-reside. Its transient joins the one
  serialized reserve. The read hint drives the engine's projected-I/O advisory only, and never refuses.
- **fp32** (parity only) is unmeasured, so it returns nil and falls through.
- **Pool cap:** the floor and peaks were measured under the engine's default MLX pool cap: `.automatic`, min(2 GB, 5 %
  of budget), so 2 GB here. A host with a larger pool would raise the idle floor.
- **Tests:** `testStagedFootprintResolvesThroughHints` resolves each configuration with the engine's own `ResolvedLane`
  and checks that the hints cover the measured idle and peaks. `testFootprintConformanceResidentAndStaged` runs the
  FIT gate on every variant × tier × residency.
- **mlx-engine side:** they rewrote the stale `residentBytesHint` doc comment to the persistent-half meaning
  (mlx-engine-swift 404176e, doc only).

## V2b zh content probe — tier C on Mandarin (2026-11-01, AB-T-0413)

**Why.** Upstream's MLX branch (`feat/mlx-apple-silicon` @ 6943a1e, `docs/MLX.md` § Quantization) found 4-bit
English-safe but Mandarin-degrading: the cookbook de-accent turned 論文…導師 into 任務…倒死. Their 4-bit covers DiT +
Thinker; our tier C quantises only the encoder. The 2026-10-18 matrix had one zh clip, an enhancement with no content to
get wrong.

**Setup.** The same pre-registered protocol (PORTING-SPEC §6), on three Mandarin content tasks:

| task | request | source / voice prompt | seconds |
|---|---|---|---|
| `zhzs` | cookbook 1.1 template, "我的导师说，这篇论文的结论还需要更多实验数据来支持。" | `zero-shot-tts/ref.wav` (the cookbook's English voice) | 6.5 |
| `zhzsn` | same template, "今天下午三点，请大家准时到会议室讨论新产品的发布计划。" | `nv/zh-a-input.wav` (native Mandarin) | 6.5 |
| `deaccent` | cookbook 4.3 verbatim, "请把方言腔改成标准普通话发音,音色维持一致。" | `accent/accent-sichuan-input.wav` | same length |

- **Reference.** The fp32 CPU oracle (`dump_goldens.py --tasks zhzs,zhzsn,deaccent`) on both variants. CER is
  scored against the oracle's own transcript, as §6 says.
- **Matrix.** Tiers R / A / B / C × {Base, Flash} × seeds {0, 1, 2}: 72 renders.
  - Binary: `auk-gates` built from 01a88e2.
  - Scorer: `score_v2b.py` in the speech-edit-eval EditX venv.
  - Diagnostic only: CER against the target text (`err_tgt`).
- Raw: `receipts/v2b-zh-{render.log,scores.csv,score.log}`.

**Result: tier C fails §6 on Mandarin content.**

| tier | CER | worst cell | SIM to source | DNSMOS | Base RTF | Flash RTF | MLX peak | §6, zh probe alone |
|---|---|---|---|---|---|---|---|---|
| R fp32 (TF32 off) | 9.01 % | — | 0.797 | 3.507 | 1.61 | 0.26 | 23.7 GB | reference |
| A bf16 | 8.97 % (−0.04) | +1.4 | 0.757 (−0.040) | 3.476 | 0.82 | 0.18 | 12.5 GB | CER pass, SIM fail |
| B bf16 + 8-bit encoder | 8.53 % (−0.48) | +1.4 | 0.754 (−0.043) | 3.484 | 0.83 | 0.12 | 9.3 GB | CER pass, SIM fail |
| C bf16 + 4-bit encoder | 12.70 % (**+3.68**) | **+14.3** (Base de-accent) | 0.764 (−0.033) | 3.463 | 0.85 | 0.11 | 7.6 GB | **FAIL** (CER and SIM) |

CER per cell (mean of 3 seeds):

| variant / task | R | A | B | C |
|---|---|---|---|---|
| Base de-accent | 3.6 % | 3.6 | 3.6 | **17.9** |
| Flash de-accent | 14.3 | 15.5 | 14.3 | **25.0** |
| Base zhzs | 10.1 | 11.6 | 11.6 | 11.6 |
| Flash zhzs | 26.1 | 23.2 | 21.7 | 21.7 |
| Base / Flash zhzsn | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

**Reading.**
- **Tier C reproduces upstream's failure, on the same words.** All 6 tier-C de-accent renders lose both
  content words:
  - 论文 becomes 论悟 / 论物 / 论务 / 嫩文;
  - 导师 becomes 盗死 / 刀死 / 叨死 / 到实.

  R, A and B keep 论文 in 18 / 18 renders. They keep 导师 in 12 / 18; the rest are the same seed-matched near-misses
  (导视 / 倒视) in every one of those tiers, R included. Within a seed, R, A and B transcripts are near-identical; C
  diverges at exactly these words. Flash de-accent is +10.7, just past the cell bound.
- **Zero-shot zh TTS is unaffected at C.** zhzsn is clean at every tier. zhzs is within ±4.4 points of R, and C
  is better than R on Flash.
  - zhzs is noisy in every tier, oracle included: the oracle itself reads 导师 as 道士, and 论文 as 文文.
  - That comes from the English voice prompt; it is model behaviour, not quantisation.
- **Hypothesis (not measured):** the 4-bit thinker damages *reading* accented content from audio (the edit path),
  not *following* text (the TTS path). The six original tasks are all English or text-led, which is why the
  2026-10-18 matrix missed it.
- **Bound.** AB-T-0413 names a "+1.5-point cell bound". That is tier C's measured worst cell from 2026-10-18; §6's
  pre-registered cell bound is +10 and the pooled bound +2.0. Tier C misses all three.
  - Resolution: one character on the 28-character de-accent sentence is 3.6 points.
  - C's Base de-accent misses five characters in every seed.
- **SIM, a second finding.** De-accent SIM to source drops 0.08–0.10 in *every* reduced-precision tier: Base
  0.699 → 0.62–0.65, Flash 0.815 → 0.72–0.74. A bf16 drops as much as C, so this is the bf16 DiT/VAE path, not the
  4-bit encoder.
  - It takes A, B and C past the −0.03 SIM bound on the probe alone.
  - Merged with the 2026-10-18 six tasks, A passes (−0.030) and B misses by 0.002 (−0.032). C still fails on its
    +14.3 cell.
  - The merged matrix is not the pre-registered one, so this is reported, not a re-verdict of A or B.
- **Not changed here.** The package manifest, the model cards and the shipped tier list are untouched. A zh caveat
  on C, or withdrawing C, is the operator's call.
- **Decision (2026-11-01, AB-D-0202):** tier C stays shipped, caveated as not for Mandarin edits. The caveat is in
  `AuKConfiguration.quant`, the speechEdit surface summary, the README and the model-card sources in `Tools/publish`.
  For Mandarin, use tier B, staged if memory is tight.

## Cookbook — upstream's verifier as the edit-kind acceptance check (AB-T-0416, 2026-11-01)

Upstream's `scripts/verify_cookbook_mlx.py` (Tencent-Hunyuan/AuK, branch `feat/mlx-apple-silicon` @ 6943a1e, MIT) is
ported as `Tools/cookbook/verify_cookbook.py`. The thresholds are upstream's:
- content edits are transcribed;
- pitch is measured in cents (+80 to +330 for +2 semitones);
- volume in dB (more than +3 dB for +10);
- speed by the length it hits (1.35× to 1.7×);
- enhancement by the noise floor, whisper by the spectral tilt;
- separation by the drop in active frames.

`auk-gates cookbook` sends each example as the request it becomes on `tts` / `speechEdit`, through
`MLXServeEngine(policy: .any)`. The run is bf16 and seed 0, with Base at its default of 32 steps × CFG 2.

Two measuring differences from upstream:
- The ASR is mlx-whisper large-v3-turbo from the HF cache, at temperature 0. Upstream used whisper-small[.en].
- Audio is read with scipy and resampled with `resample_poly`. Upstream used soxr VHQ.

Receipt: `receipts/cookbook.txt`. Renders: `WIP/auk-port/cookbook/`.

**Not run (5 of 17):**

| example | why |
|---|---|
| 2.2 lyric edit | excluded, a known model limit: it fails in the torch reference too at the same steps and seed |
| 4.4 add nonverbal | not expressible: it puts the cough before 'We tested', but `.addNonverbal` only adds at the start or end of the take |
| 5.2 speech separation | not expressible: MLXAuK declares no `audioSeparation` surface |
| 5.3 music separation | not expressible: no `audioSeparation` surface |
| 5.4 target speaker extraction | not expressible: no `audioSeparation` surface |

**The 12 expressible examples, after the two template fixes below:**

| example | request | Flash, 4 steps | Base, 32 steps |
|---|---|---|---|
| 1.1 zero-shot TTS | tts, reference clip, 6.0 s | PASS: word accuracy 100% | PASS: 100% |
| 1.2 instruct TTS | tts, `.textDescription` (Mandarin), 1.7 s | PASS: "Welcome home. How was work today?" | PASS: same |
| 2.1 content edit | `.replaceWords`, 7.0 s | PASS: new phrase in, old phrase gone | PASS |
| 3.1 pitch | `.pitch(+2)` | PASS: 190 → 217 Hz = +234 cents | PASS: +258 cents |
| 3.2 speed | `.speed(1.5)`, 6.86 s | PASS: 11.0 → 6.9 s = 1.60× | PASS: 1.60× |
| 3.3 volume | `.volume(+10)` | PASS: −26.0 → −19.1 dBFS = +6.9 dB | PASS: +9.2 dB |
| 4.1 emotion | `.emotion("happy")` | PASS: 91% of words kept, F0 +264 cents | PASS: 86% kept, +693 cents |
| 4.2 timbre | `.timbre("a deep, calm male voice")` | PASS: F0 −907 cents | PASS: −855 cents |
| 4.3 de-accent | `.deAccent` (EN template) | PASS: same length, transcribes | PASS |
| 4.4 remove nonverbal | `.removeNonverbal("humming")`, 22.0 s | PASS: 23.9 → 22.0 s | PASS |
| 4.5 whisper | `.style("whisper")` (now the CN template) | PASS: HF/LF 0.073 → 0.092 | PASS: 0.073 → 0.545 |
| 5.1 enhancement | `.denoise` (approximates the combined enhance) | PASS: noise floor −37.4 dB | PASS: −34.6 dB |
| **cookbook total** | | **12/12** | **12/12** |

**The int8 and int4 tiers.** These are the encoder tiers MLXAuK declares; the DiT and VAE stay bf16. Both pass
12/12 on both variants, and the supplementary rows pass 4/4. The figures move a little:
- Base whisper HF/LF is 0.587 at int8 and 0.396 at int4, against 0.545 at bf16.
- Flash volume is +6.8 dB at int4, against +6.9 dB at bf16.
- Pitch is unchanged: Flash +234 cents, Base +258.

The Mandarin probe at 4 bits (upstream's "論文…導師" → "任務…倒死") is AB-T-0413's lane, not this one. This one
can't see it: see the de-accent note below.

**Against upstream** (Flash 15/17, Base＠32 16/17). Upstream fails only 2.2 on Base, and 2.2 plus one more on Flash.
Upstream does not name Flash's second failure, but its write-up says Flash's whisper ratio opens an order of
magnitude, so it isn't 4.5. Of the 12 examples both ports can run, we pass all 12 on both variants: no example fails
here that passes upstream. The 4 examples we cannot express are separation and the inline nonverbal add; 2.2 is
excluded.

Upstream's volume figures are +7.4 dB on Flash and +9.7 dB on Base. Ours are +6.9 and +9.2 dB.

**Supplementary rows: the declared kinds with no cookbook example.**

These run on the cookbook's own inputs and are reported apart from the count.

| row | request | Flash | Base |
|---|---|---|---|
| insert | `.insertWords("quietly", before: "accepting")`, length estimated by the package | PASS: the word is inserted and the anchor kept, 7.5 s | PASS |
| remove | `.removeWords("but accepting what we cannot have")` | PASS: phrase gone, 7.0 → 4.7 s | PASS |
| add nonverbal at the start | `.addNonverbal("cough", atEnd: false)`, 14.3 s | PASS: 100% of words kept, 14.3 s | PASS |
| dereverb | `.dereverberate` | PASS: same length, still transcribes | PASS |
| whisper → normal | `.style("normal")` | information only | information only |

Whisper → normal isn't judged. The assets' only whisper sample, `wh-w2n-zh-input`, tilts like plain speech: its HF/LF
is 0.073, and the EN take's is 0.057. So the assets hold no whispered input to bring back to a normal voice.

**Two package bugs, found and fixed here.**

At v0.2.0 (`01a88e2`), Flash passed 10/12 and Base passed 12/12. Both Flash failures came from the package's own
templates, which the earlier lanes never ran. A seed sweep, seeds 0–4, through `auk-gates render`
(`Tools/cookbook/sweep_phrasing.sh`), split the template effect from seed noise:

| phrasing | Flash, seeds passing | Base, seeds passing |
|---|---|---|
| 1.2, v0.2.0 template ("Generate speech based on the following description: … The content to speak is: …") | 1/5 | 4/5 |
| 1.2, the cookbook example's phrasing, which the V0 oracle also used | 5/5 | 5/5 |
| 4.5 on the zh sample, EN template | 0/5: HF/LF falls, 0.062–0.067 | 5/5, but 0.080–0.082 is a 10% move |
| 4.5 on the zh sample, CN template (the cookbook example) | 5/5: 0.080–0.097 | 5/5: 0.26–6.1 |
| to-whisper on an EN take, EN template | 2/5: 0.054–0.058 against a source of 0.057 | 1/5 |
| to-whisper on an EN take, CN template | 3/5: up to 0.096 | 5/5: 0.19–15 |

The fixes:
- `AukInstruction.instructTTS` EN text is now the cookbook example's phrasing.
- `AuKSurfaces.language(for:)` sends `.style("whisper")` to the CN template, whatever the take's language.

Both are pinned by assertions in the existing `testInstructionTemplates` and whisper tests. `swift test`: 21/21.
Flash to-whisper is still weak on an English take (3/5).

**Notes:**
- 4.3 de-accent passes, but the check is shallow: same length, and the output transcribes. The Sichuan source itself
  transcribes as "…伦文婷告发给倒死…", and both edits as "…任文婷告发给倒死…". The garbled 论文 / 导师 come from the
  ASR on this speaker, not from the edit. So this lane can't show whether the accent moved.
- Render time for the whole set: Flash 13 s and Base 95 s for 17 renders, about 7× apart.

## audioSeparation — the third surface (2026-11-01)

`MLXAuK` now serves `audioSeparation` (contract 1.59.0). Stems: `.speaker`, with a speaker pick `.ordinal` (the n-th
to start talking) or `.saying` (the one who says given words); `.singing`; and `.voices` on Base only. Each stem is one
AuK run that regenerates that source at the mixture's length. So no stem is derived by subtraction, and `.instrumental`
or `.music` are refused. Raw outputs: `receipts/separation.txt`.

**Checks.** `ss/en-1` is a two-person dialogue whose lines are known: speaker A opens and asks, speaker B answers
("get what", "banking and finance"). So a kept speaker is judged by WHICH lines whisper hears in it. It passes with at
least 2 of the right speaker's lines and none of the other's; the mixture itself hears A 5/5 and B 4/4. The cookbook's
`zh-1` (Cantonese, many speakers) keeps upstream's check, the drop in voiced frames. Singing must keep the lyric and
lose the backing below 200 Hz.

**Phrasing sweep** (`auk-gates render`, bf16, seeds 0–2; passes out of 3). "tpl" is the cookbook's template line;
"example" is its CLI example's phrasing:

| ask | Flash, tpl | Flash, example | Base, tpl | Base, example |
|---|---|---|---|---|
| first speaker (`.ordinal(1)`) | 2 | **3** | 2 | **3** |
| second speaker (`.ordinal(2)`) | **1** | 0, invented speech | **3** | 1 |
| who says "get what" (`.saying`) | 3 | — | 2 (seed 1 kept both) | — |
| singing only | 3 | 3 | 2 (seed 1 kept the bass) | **3** |
| all voices | 1, invented lyrics on 2 | — | 3 | — |

Choices made from it:
- **Speaker picks keep the template.** The example's phrasing helps the first speaker, but it makes Flash invent speech
  for the second, and that is the worse failure.
- **Singing takes the example's phrasing** ("Keep the clean singing voice, drop all other audio.").
- **Flash does not declare `.voices`.** It refuses the stem and points to `.singing` or Base.
- **Flash's surface summary warns that a later speaker by order is unreliable,** and steers to `.saying`.

**Cookbook through MLXServeEngine** (seed 0, `Tools/cookbook/run_cookbook.sh`):
- **Flash and Base each pass 15/15 expressible examples**, up from 12/12, now that 5.2–5.4 run. 2.2 is excluded and
  4.4-add is still unexpressible.
- Supplementary rows: Flash 6/6 (`S-voices` not offered), Base 7/7.
- 5.4 keeps B (A 0/5, B 3/4) on both variants.
- The first-speaker row keeps A 5/5 with no B line; the second-speaker row keeps B (2/4) with no A line.
- Every earlier row still passes.

**Not measured:**
- the int8 and int4 tiers on separation;
- seeds beyond 0–2;
- whether the kept Cantonese speaker in `zh-1` is the right one: no ground truth, only fewer voiced frames;
- speaker order beyond the second.
