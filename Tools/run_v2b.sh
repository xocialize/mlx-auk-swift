#!/bin/bash
# V2b render matrix (PORTING-SPEC §6): tiers R/A/B/C/D × {flash, base} × 6 tasks × seeds 0,1,2. Resumable: `matrix`
# skips renders whose wav exists. Score afterwards with Tools/asr/score_v2b.py <out-dir> <goldens> <assets> <csv>.
#
#   [TIERS="R A B C"] [AUK_GATES=<binary>] Tools/run_v2b.sh <goldens> <weights> <assets> [out-dir]
#     goldens  oracle output root: <goldens>/{base,flash}/<task>/meta.json (the requests)
#     weights  converter output root: AuK[-Flash]-{fp32,bf16}/, Qwen2.5-Omni-3B-Thinker-{fp32,bf16}/
#     assets   the upstream clone's assets/demo-input-audio (the source clips the metas name)
#     out-dir  render root (default: ./v2b)
#     TIERS    subset of tiers to run (default: all five); AUK_GATES overrides the built binary
set -uo pipefail
[ $# -ge 3 ] || { sed -n 5,10p "$0"; exit 2; }
G=$1; W=$2; A=$3; O=${4:-v2b}
B=${AUK_GATES:-"$(cd "$(dirname "$0")/.." && pwd)/.build/out/Products/Release/auk-gates"}
T=" ${TIERS:-R A B C D} "
[ -x "$B" ] || { echo "build first: swift build -c release --build-system swiftbuild --product auk-gates"; exit 1; }
for v in flash base; do
  m32=$([ $v = flash ] && echo AuK-Flash-fp32 || echo AuK-fp32)
  m16=$([ $v = flash ] && echo AuK-Flash-bf16 || echo AuK-bf16)
  if [[ $T == *" R "* ]]; then echo "== R $v"; MLX_ENABLE_TF32=0 "$B" matrix "$G/$v" "$A" "$W/$m32" "$W/Qwen2.5-Omni-3B-Thinker-fp32" "$O/R/$v"; fi
  if [[ $T == *" A "* ]]; then echo "== A $v"; "$B" --dtype bfloat16 matrix "$G/$v" "$A" "$W/$m16" "$W/Qwen2.5-Omni-3B-Thinker-bf16" "$O/A/$v"; fi
  if [[ $T == *" B "* ]]; then echo "== B $v"; "$B" --dtype bfloat16 --enc-bits 8 matrix "$G/$v" "$A" "$W/$m16" "$W/Qwen2.5-Omni-3B-Thinker-bf16" "$O/B/$v"; fi
  if [[ $T == *" C "* ]]; then echo "== C $v"; "$B" --dtype bfloat16 --enc-bits 4 matrix "$G/$v" "$A" "$W/$m16" "$W/Qwen2.5-Omni-3B-Thinker-bf16" "$O/C/$v"; fi
  if [[ $T == *" D "* ]]; then echo "== D $v"; "$B" --dtype bfloat16 --dit-bits 8 --enc-bits 8 matrix "$G/$v" "$A" "$W/$m16" "$W/Qwen2.5-Omni-3B-Thinker-bf16" "$O/D/$v"; fi
done
