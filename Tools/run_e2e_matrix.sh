#!/bin/bash
# V2 end-to-end parity matrix: every golden task × {base, flash} through auk-gates e2e, GPU fp32 with TF32 off
# (MLX's M5 default TF32 costs ~1e-3 relL2 per GEMM — AB-D-0114; the exact lane needs it off).
set -u
cd "$(dirname "$0")/.."
export MLX_ENABLE_TF32=0
G=auk-port/oracle/goldens; W=auk-port/weights; B=../PROD/AuK/mlx-auk-swift/.build/out/Products/Release/auk-gates
OUT=auk-port/receipts/e2e-fp32-gpu-tf32off.txt; : > "$OUT"
for v in flash base; do
  m=$([ $v = flash ] && echo AuK-Flash-fp32 || echo AuK-fp32)
  for t in zs itts content speed emotion se; do
    echo "== $v/$t" | tee -a "$OUT"
    "$B" e2e $G/$v/$t $W/$m $W/Qwen2.5-Omni-3B-Thinker-fp32 auk-port/swift-out/$v-$t-fp32.wav 2>&1 | grep -E "max\||wall|ERROR" | tee -a "$OUT"
  done
done
