#!/bin/bash
# Staged residency (AB-T-0414): bit identity against resident mode, the per-call reload cost, and phys_footprint peaks.
#
#   Tools/run_staged.sh bit <goldens> <weights> <assets> [out-dir]
#       `auk-gates staged` per variant × tier: every golden task twice (production path, oracle noise) through a
#       resident then a staged pipeline; staged must be bit-identical. bf16 also runs a cold-page-cache staged pass.
#   Tools/run_staged.sh mem <goldens> <weights> <assets> [out-dir]
#       `auk-gates validate` per variant × tier × residency: the engine lane, with a 5 ms phys_footprint sampler and
#       the kernel's lifetime peak. One process per run, so neither mode inherits the other's process memory.
#
#     goldens  oracle output root: <goldens>/{base,flash}/<task>/
#     weights  converter output root: AuK[-Flash]-bf16/, Qwen2.5-Omni-3B-Thinker-{bf16,8bit,4bit}/
#     assets   the upstream clone's assets/demo-input-audio
#     out-dir  logs and renders (default: ./staged)
# AUK_GATES overrides the binary (default: this checkout's release build).
set -uo pipefail
[ $# -ge 4 ] || { sed -n 4,16p "$0"; exit 2; }
P=$1; G=$2; W=$3; A=$4; O=${5:-staged}
B=${AUK_GATES:-"$(cd "$(dirname "$0")/.." && pwd)/.build/out/Products/Release/auk-gates"}
[ -x "$B" ] || { echo "build first: swift build -c release --build-system swiftbuild --product auk-gates"; exit 1; }
mkdir -p "$O"
enc() { case $1 in bf16) echo Qwen2.5-Omni-3B-Thinker-bf16 ;; int8) echo Qwen2.5-Omni-3B-Thinker-8bit ;; int4) echo Qwen2.5-Omni-3B-Thinker-4bit ;; esac; }
for v in flash base; do
  m=$([ $v = flash ] && echo AuK-Flash-bf16 || echo AuK-bf16)
  for q in bf16 int8 int4; do
    case $P in
      bit)
        cold=$([ $q = bf16 ] && echo --cold || true)
        echo "== staged $v $q"
        "$B" --dtype bfloat16 $cold staged "$G/$v" "$A" "$W/$m" "$W/$(enc $q)" "$O/bit-$v-$q" 2>&1 | tee "$O/staged-$v-$q.log" ;;
      mem)
        for r in resident staged; do
          echo "== validate $v $q $r"
          "$B" --residency $r validate $v $q "$W/$m" "$W/$(enc $q)" "$A" "$O/val-$v-$q-$r" 2>&1 | tee "$O/validate-$v-$q-$r.log"
        done ;;
    esac
  done
done
