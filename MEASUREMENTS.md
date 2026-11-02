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
