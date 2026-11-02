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
  idle. V3's after-request readings showed 0.80–1.03 GB. So the declared `peakActivationBytes` of 1.5 GB is short by
  up to ~0.75 GB in total: base int4 measures 10.04 GB against 7.8 + 1.5 = 9.3 GB, and base bf16 14.92 against 14.4. This is not changed here; it is a follow-up.

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

**Declaration.** The ask is AB-A-0246 to mlx-engine, still open at commit time. The proposal is the existing
`FootprintConfigured` hints with no contract change:
- `residentBytesHint` = the staged idle (~3.0 GB);
- `peakActivationBytesHint` = staged peak − idle (bf16 8.0, int8 5.0, int4 4.6 GB);
- `expectedWeightReadBytesPerRunHint` = encoder + DiT bytes.

Until mlx-engine answers, staged ships as **package configuration only**, without `FootprintConfigured`. The engine
keeps charging the resident footprints (12.9 / 9.6 / 7.8–8.0 GB + 1.5 GB). This over-reserves a staged registration,
which is safe. It also means the engine does not yet admit staged AuK on a budget that only the staged numbers fit.
