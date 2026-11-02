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
