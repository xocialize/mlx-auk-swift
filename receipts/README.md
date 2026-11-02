# Receipts

The raw outputs behind `MEASUREMENTS.md`, from an M5 Max with mlx-swift 0.32.3, recorded 2026-10-18.

| file | what | produced by |
|---|---|---|
| `e2e-fp32-gpu-tf32off.txt` | V2: the end-to-end parity matrix, 12 lanes against the fp32 oracle | `Tools/run_e2e_matrix.sh` |
| `v2b-render.log` | V2b: per-render wall time and MLX peak, tiers R/A/B/C/D | `Tools/run_v2b.sh` |
| `v2b-scores.csv` | V2b: per render, the ASR transcript, WER/CER, CAM++ SIM to source, DNSMOS | `Tools/asr/score_v2b.py` |
| `v2b-score.log` | V2b: the scorer's log, pooled means and the pass/fail verdict per tier | `Tools/asr/score_v2b.py` |
| `validate.log` | V3: MLXServeEngine register → prepare → run → evict, phys_footprint per tier × variant | `auk-gates validate` |
| `validate-1.59.log` | contract 1.59.0: the same engine lane plus replace, speed, pitch and remove-nonverbal edits and an off-grid refusal, bf16 × variant (2026-11-01) | `auk-gates validate` |
| `stage-profile.log` | AB-T-0415: per-stage wall clock (ref / encoder / ODE / decode), warm, 5 runs per task, Flash and Base × bf16 and 8-bit encoder (2026-11-01) | `Tools/run_profile.sh` → `auk-gates profile` |
| `staged-residency.txt` | AB-T-0414: staged vs resident — bit identity (2 variants × 3 tiers × 12 requests, bf16 also cold page cache), per-request reload cost, and engine-lane phys_footprint peaks (5 ms sampler + kernel ledger), 12 processes; plus the page-cache eviction probe (2026-11-01) | `Tools/run_staged.sh bit\|mem` |
| `hf-upload.log` | V4: the Hugging Face commit for each published repo | `hf upload` |
