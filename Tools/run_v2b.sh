#!/bin/bash
# V2b render matrix (PORTING-SPEC §6): tiers R/A/B/C/D × {flash, base} × 6 tasks × seeds 0,1,2.
# Resumable: matrix skips renders whose wav exists.
set -u
AP=/Volumes/Satechi/Development/mlxengine-audio/WIP/auk-port
B=/Volumes/Satechi/Development/mlxengine-audio/PROD/AuK/mlx-auk-swift/.build/out/Products/Release/auk-gates
G=$AP/oracle/goldens; W=$AP/weights; A=$AP/ref/AuK/assets/demo-input-audio; O=$AP/v2b
for v in flash base; do
  m32=$([ $v = flash ] && echo AuK-Flash-fp32 || echo AuK-fp32)
  m16=$([ $v = flash ] && echo AuK-Flash-bf16 || echo AuK-bf16)
  echo "== R $v";  MLX_ENABLE_TF32=0 "$B" matrix $G/$v $A $W/$m32 $W/Qwen2.5-Omni-3B-Thinker-fp32 $O/R/$v
  echo "== A $v";  "$B" --dtype bfloat16 matrix $G/$v $A $W/$m16 $W/Qwen2.5-Omni-3B-Thinker-bf16 $O/A/$v
  echo "== B $v";  "$B" --dtype bfloat16 --enc-bits 8 matrix $G/$v $A $W/$m16 $W/Qwen2.5-Omni-3B-Thinker-bf16 $O/B/$v
  echo "== C $v";  "$B" --dtype bfloat16 --enc-bits 4 matrix $G/$v $A $W/$m16 $W/Qwen2.5-Omni-3B-Thinker-bf16 $O/C/$v
  echo "== D $v";  "$B" --dtype bfloat16 --dit-bits 8 --enc-bits 8 matrix $G/$v $A $W/$m16 $W/Qwen2.5-Omni-3B-Thinker-bf16 $O/D/$v
done
