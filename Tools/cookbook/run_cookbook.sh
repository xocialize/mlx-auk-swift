#!/bin/zsh
# run_cookbook.sh — every docs/COOKBOOK.md example MLXAuK can express, through MLXServeEngine, on Base and Flash,
# then the cookbook verifier over both (AB-T-0416).
#
#   run_cookbook.sh <weights> <assets> <out> [quant=bf16] [seed=0] [variants="flash base"]
#
#   <weights>  holds AuK-bf16/, AuK-Flash-bf16/ and Qwen2.5-Omni-3B-Thinker-{bf16,8bit,4bit}/ (Tools/convert)
#   <assets>   upstream's assets/demo-input-audio
#   <out>      renders land in <out>/<variant>-<quant>/ ; the verifier's table in <out>/verify-<quant>.txt / .json
#
# Env: AUK_GATES (the release auk-gates binary; default .build/release/auk-gates), AUK_PY (a Python with numpy,
# scipy and mlx-whisper; default python3). Set AUK_LOCK to a command prefix (e.g. a GPU-lock wrapper) to run each
# render and the scoring pass under it; it is called as `$AUK_LOCK <label> <command…>`.
set -e -o pipefail
weights=$1 assets=$2 out=$3 quant=${4:-bf16} seed=${5:-0} variants=(${=${6:-flash base}})
here=${0:A:h}
gates=${AUK_GATES:-$here/../../.build/release/auk-gates}
py=${AUK_PY:-python3}
locked() { local label=$1; shift; if [[ -n ${AUK_LOCK:-} ]]; then ${=AUK_LOCK} $label "$@"; else "$@"; fi }
enc=$weights/Qwen2.5-Omni-3B-Thinker-${${quant/int8/8bit}/int4/4bit}
mkdir -p $out; dirs=()
for v in $variants; do
    model=$weights/$([[ $v == base ]] && echo AuK || echo AuK-Flash)-bf16
    d=$out/$v-$quant
    locked cookbook-$v-$quant $gates cookbook $v $quant $model $enc $assets $d $seed 2>&1 | tee $out/render-$v-$quant.log
    dirs+=($d)
done
locked cookbook-verify-$quant $py -I $here/verify_cookbook.py $assets $dirs --json $out/verify-$quant.json \
    2>&1 | tee $out/verify-$quant.txt
