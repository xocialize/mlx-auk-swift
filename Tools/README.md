# Tools

The Python and shell side of the port: the fp32 torch oracle, the weight converter, the scorers, the run matrices
and the publish staging. Nothing here assumes a directory layout: every location is an argument.

| path | what |
|---|---|
| `oracle-capture/dump_goldens.py` | **V0 oracle.** Runs upstream AuK (`Tencent-Hunyuan/AuK@9481474`) in fp32 on the CPU and dumps per-stage goldens for 6 cookbook tasks (default) or the 3 zh content-probe tasks `zhzs,zhzsn,deaccent` (AB-T-0413). Fixed seeds, so the output is bit-reproducible. `--variant base\|flash --assets <clone>/assets/demo-input-audio --out <goldens> [--tasks …]` |
| `convert/convert.py` | **V1 converter.** `tencent/AuK`, `tencent/AuK-Flash` and `Qwen/Qwen2.5-Omni-3B` (read from the HF cache) → MLX-layout bundles: weight-norm folded, convs transposed, flow / vision / talker dropped. `--out <weights> [--dtype float32\|bfloat16] [--parts base,flash,encoder]` |
| `convert/gate_v1.py` | **V1 gate.** The converted tensors must equal upstream's own `remove_weight_norm` output. `gate_v1.py <weights>/AuK-fp32` |
| `run_e2e_matrix.sh` | **V2 matrix.** Every golden through `auk-gates e2e` on the GPU, fp32 with `MLX_ENABLE_TF32=0`. `run_e2e_matrix.sh <goldens> <weights> [out]` |
| `run_v2b.sh` | **V2b matrix.** Tiers R / A / B / C / D × every golden task × 3 seeds through the production path. `[TIERS="R A B C"] [AUK_GATES=<bin>] run_v2b.sh <goldens> <weights> <assets> [out]` |
| `run_profile.sh` | **Stage profile** (AB-T-0415). `auk-gates profile` over the 6 goldens for Flash and Base × bf16 and 8-bit encoder: 1 warm-up + 5 timed runs per task, wall clock per stage (ref / encoder / ODE / decode). Run it with the GPU to itself. `run_profile.sh <goldens> <weights> <assets> [out] [runs]` |
| `run_staged.sh` | **Staged residency (AB-T-0414).** `bit`: `auk-gates staged` per variant × tier, staged must equal resident bit for bit (bf16 also cold page cache). `mem`: `auk-gates validate --residency resident\|staged` per variant × tier. `run_staged.sh bit\|mem <goldens> <weights> <assets> [out]` |
| `asr/score_v2b.py` | **V2b scorer.** mlx-whisper WER/CER, CAM++ SIM and DNSMOS, plus the pre-registered pass rule; an earlier CSV as a 5th argument adds a merged verdict. `AUK_FUNASR_DETACH_ROOT=<Step-Audio-EditX clone> score_v2b.py <v2b> <goldens> <assets> <out.csv> [earlier.csv]` |
| `asr/transcribe.py` | Quick content check with mlx-whisper large-v3-turbo. |
| `cookbook/run_cookbook.sh` | **Cookbook acceptance (AB-T-0416).** Every `docs/COOKBOOK.md` example MLXAuK's surfaces can express, through `MLXServeEngine` (`auk-gates cookbook`), on Base and Flash, then the verifier over both. `run_cookbook.sh <weights> <assets> <out> [quant] [seed] [variants]`; `AUK_LOCK` wraps each render and the scoring pass |
| `cookbook/verify_cookbook.py` | **Cookbook verifier.** Upstream's `scripts/verify_cookbook_mlx.py` (feat/mlx-apple-silicon @ 6943a1e, MIT, notice in `cookbook/LICENSE-upstream-AuK`) ported to `auk-gates cookbook` output: content transcribed, pitch in cents, volume in dB, speed by duration, enhancement by noise floor, separation by active frames. Unexpressible examples are listed, not counted; `supplementary` rows cover declared kinds with no cookbook example |
| `cookbook/sweep_phrasing.sh`, `cookbook/score_sweep.py` | Instruction phrasings × seeds 0–4 through `auk-gates render`, scored with the verifier's checks. They separate a template effect from seed noise, and are how the instruct-TTS and to-whisper template fixes were found |
| `publish/stage.sh` | **V4.** Stages each bundle's LICENSE, NOTICE and model card (sources in `publish/auk`, `publish/qwen-thinker`) before `hf upload`. |

Swift-side lanes are `auk-gates` subcommands: `vae`, `dit`, `encoder`, `e2e`, `matrix`, `validate`, `cookbook`,
`quantize-encoder`, `render`, `profile` and `staged` (header of `Sources/auk-gates/main.swift`).

**Environment.** The oracle was validated with torch 2.7.1, torchaudio 2.7.1, transformers 4.57.1, x_transformers
2.31.8, qwen-omni-utils (+ audioread, av), omegaconf, torchdiffeq and accelerate, plus `pip install -e <AuK clone>
--no-deps`. Run the venv's Python with `-I`.

**Full rebuild from nothing:**
1. Run `dump_goldens.py` for both variants.
2. Run `convert.py` at fp32 and at bf16.
3. Run `auk-gates quantize-encoder` for 8 and 4 bits.
4. Run `run_e2e_matrix.sh`.
5. Run `run_v2b.sh`, then `score_v2b.py`.
6. Run `auk-gates validate` for each variant × tier.

The raw outputs of the 2026-10-18 run are in `../receipts/`.

Build the gate CLI with the Xcode build system so the MLX metallib ships (AB-B-0007):
`swift build -c release --build-system swiftbuild --product auk-gates`.
