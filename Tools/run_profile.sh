#!/bin/bash
# AB-T-0415 stage profile: `auk-gates profile` over the 6 golden tasks for {Flash, Base} × {bf16, 8-bit encoder}
# (shipped tiers A and B), 1 warm-up + 5 timed runs per task. Each config is a fresh process, run back to back.
#
#   Tools/run_profile.sh <goldens> <weights> <assets> [out-dir] [runs]
#     goldens  oracle output root: <goldens>/{base,flash}/<task>/meta.json (the requests)
#     weights  converter output root: AuK[-Flash]-bf16/, Qwen2.5-Omni-3B-Thinker-{bf16,8bit}/
#     assets   the upstream clone's assets/demo-input-audio
#     out-dir  where <config>.log and <config>.jsonl go (default: ./profile)
set -uo pipefail
[ $# -ge 3 ] || { sed -n 5,9p "$0"; exit 2; }
G=$1; W=$2; A=$3; O=${4:-profile}; N=${5:-5}
B="$(cd "$(dirname "$0")/.." && pwd)/.build/out/Products/Release/auk-gates"
[ -x "$B" ] || { echo "build first: swift build -c release --build-system swiftbuild --product auk-gates"; exit 1; }
mkdir -p "$O"
for v in flash base; do
  m=$([ $v = flash ] && echo AuK-Flash-bf16 || echo AuK-bf16)
  for e in bf16 8bit; do
    c=$v-$([ $e = bf16 ] && echo bf16 || echo int8)
    echo "== $c"
    rm -f "$O/$c.jsonl"
    "$B" --dtype bfloat16 profile "$G/$v" "$A" "$W/$m" "$W/Qwen2.5-Omni-3B-Thinker-$e" "$O/$c.jsonl" "$N" 2>&1 | tee "$O/$c.log"
  done
done
