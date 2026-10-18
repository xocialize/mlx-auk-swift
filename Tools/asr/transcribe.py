"""ASR check: transcribe wavs with mlx-whisper large-v3-turbo (the cached mlx-community copy)."""
import sys, json
import mlx_whisper
out = {}
for p in sys.argv[1:]:
    r = mlx_whisper.transcribe(p, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
    out[p] = r["text"].strip()
    print(f"{p.split('goldens/')[-1]}: {out[p]}")
