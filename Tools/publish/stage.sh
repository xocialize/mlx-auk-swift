#!/bin/bash
# Stage the licence bundle and model card into converted bundle directories before `hf upload`.
# AuK repos get Tencent's MIT LICENSE; Thinker repos get the Qwen RESEARCH LICENSE, the NOTICE §3c requires, and
# a card stating the files are modified, plus "Built with Qwen" (§4b).
#
#   Tools/publish/stage.sh <weights>     # stages AuK-bf16, AuK-Flash-bf16, Qwen2.5-Omni-3B-Thinker-{bf16,8bit,4bit}
#   hf upload mlx-community/<repo> <weights>/<repo> . --repo-type model
set -euo pipefail
[ $# -eq 1 ] || { sed -n 6,7p "$0"; exit 2; }
W=$1; H="$(cd "$(dirname "$0")" && pwd)"; L="$H/../../LICENSES"
for V in AuK AuK-Flash; do
  cp "$L/MIT-tencent-auk.txt" "$W/$V-bf16/LICENSE"
  cp "$H/auk/README-$V.md" "$W/$V-bf16/README.md"
done
for T in bf16 8bit 4bit; do
  D="$W/Qwen2.5-Omni-3B-Thinker-$T"; [ -d "$D" ] || { echo "skip $D (not converted)"; continue; }
  cp "$L/Qwen-Research-License-Agreement.txt" "$D/LICENSE"; cp "$H/qwen-thinker/NOTICE" "$D/"
  case $T in
    bf16) Q="bfloat16, unquantised (7.45 GB).";;
    8bit) Q="The 36 LLM layers and 32 audio-tower layers have their Linears quantised to affine 8-bit, group size 64 (4.26 GB). Embeddings, norms and projections are bf16.";;
    4bit) Q="The 36 LLM layers and 32 audio-tower layers have their Linears quantised to affine 4-bit, group size 64 (2.55 GB). Embeddings, norms and projections are bf16.";;
  esac
  sed -e "s/__TIER__/$T/; s|__QUANT__|$Q|" "$H/qwen-thinker/README.md" > "$D/README.md"
done
echo "staged under $W"
