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
| `cookbook.txt` | AB-T-0416: every cookbook example MLXAuK can express, through MLXServeEngine, Base and Flash, seed 0 — render logs and the verifier at bf16 after the template fixes and at v0.2.0, then the int8 and int4 tiers; the phrasing seed sweeps; the de-accent ASR on the source (2026-11-01) | `Tools/cookbook/run_cookbook.sh`, `sweep_phrasing.sh` + `score_sweep.py` |
| `hf-upload.log` | V4: the Hugging Face commit for each published repo | `hf upload` |
