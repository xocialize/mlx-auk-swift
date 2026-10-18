# Tools

The Python side of the port: the fp32 torch oracle, the weight converter, and the checks run against them.
Each script runs from a venv outside this repo; the commands below are the ones the gates were run with.

| path | what |
|---|---|
| `oracle-capture/dump_goldens.py` | V0 oracle. Runs upstream AuK (`Tencent-Hunyuan/AuK@9481474`) in fp32 on the CPU and dumps per-stage goldens for 6 cookbook tasks. Venv: torch 2.7.1, transformers 4.57.1, x_transformers 2.31.8, qwen-omni-utils, `pip install -e <AuK clone> --no-deps`. |
| `convert/convert.py` | V1 converter. `tencent/AuK`, `tencent/AuK-Flash` and `Qwen/Qwen2.5-Omni-3B` → MLX-layout safetensors (weight-norm folded, convs transposed, flow/vision/talker dropped), fp32 or bf16. |
| `convert/gate_v1.py` | V1 gate. The converted tensors must equal upstream's own `remove_weight_norm` output. |
| `run_e2e_matrix.sh` | V2 end-to-end matrix: every golden through `auk-gates e2e` on the GPU, fp32 with `MLX_ENABLE_TF32=0`. |
| `asr/transcribe.py` | Content check with mlx-whisper large-v3-turbo. |

Build the gate CLI with the Xcode build system so the MLX metallib ships (AB-B-0007):
`swift build -c release --build-system swiftbuild --product auk-gates`.
