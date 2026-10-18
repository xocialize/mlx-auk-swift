#!/bin/bash
# V2 end-to-end parity matrix: every golden task × {base, flash} through `auk-gates e2e`, GPU fp32 with TF32 off.
# (MLX's M5 default TF32 costs ~1e-3 relL2 per GEMM — AB-D-0114; the exact lane needs it off.)
#
#   Tools/run_e2e_matrix.sh <goldens> <weights> [out-dir]
#     goldens  oracle output root (Tools/oracle-capture/dump_goldens.py): <goldens>/{base,flash}/<task>/
#     weights  converter output root (Tools/convert/convert.py): AuK-fp32/, AuK-Flash-fp32/, Qwen2.5-Omni-3B-Thinker-fp32/
#     out-dir  where the log and wavs go (default: ./e2e-out)
set -euo pipefail
[ $# -ge 2 ] || { sed -n 4,8p "$0"; exit 2; }
G=$1; W=$2; O=${3:-e2e-out}
B="$(cd "$(dirname "$0")/.." && pwd)/.build/out/Products/Release/auk-gates"
[ -x "$B" ] || { echo "build first: swift build -c release --build-system swiftbuild --product auk-gates"; exit 1; }
export MLX_ENABLE_TF32=0
mkdir -p "$O"; LOG="$O/e2e-fp32-gpu-tf32off.txt"; : > "$LOG"
for v in flash base; do
  m=$([ $v = flash ] && echo AuK-Flash-fp32 || echo AuK-fp32)
  for t in zs itts content speed emotion se; do
    echo "== $v/$t" | tee -a "$LOG"
    "$B" e2e "$G/$v/$t" "$W/$m" "$W/Qwen2.5-Omni-3B-Thinker-fp32" "$O/$v-$t-fp32.wav" 2>&1 | grep -E "max\||wall|ERROR" | tee -a "$LOG"
  done
done
