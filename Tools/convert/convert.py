"""V1 converter: upstream AuK / AuK-Flash + Qwen2.5-Omni-3B → MLX-layout safetensors for mlx-auk-swift.

Upstream key names are kept verbatim (isomorphic port), with only these changes:
  * weight_norm folded: (weight_g, weight_v) → weight, via torch._weight_norm with the dim each module used
    (dim 0 for Conv1d and for ConvTranspose1d, matching the checkpoint's weight_g shapes);
  * conv layouts transposed to MLX: Conv1d (O, I/g, K) → (O, K, I/g); ConvTranspose1d (I, O, K) → (O, K, I);
  * dropped: VAE `flow.*` (training-only coupling flow), thinker `visual` / `lm_head`, `audio_bos_eos_token`
    (unused on the text-token bos/eos path), and the whole talker / token2wav;
  * kept: the alias-free `filter` buffers (exact kaiser-sinc taps), `global_mean` / `global_log_std`,
    `layer_weights` / `layer_scale`, the DiT `rotary_embed.inv_freq`.

Outputs (per variant, under --out):
  AuK[-Flash]-<dtype>/dit.safetensors       transformer.* + layer_weights + layer_scale
  AuK[-Flash]-<dtype>/vae.safetensors       BigVGAN-Flow VAE, inference subset
  Qwen2.5-Omni-3B-Thinker-<dtype>/encoder.safetensors   thinker.model.* + thinker.audio_tower.*
plus config.json for each and the tokenizer / processor files beside the encoder.

    .venv/bin/python convert.py --out <dir> [--dtype float32|bfloat16] [--parts base,flash,encoder]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file


HF = os.path.expanduser("~/.cache/huggingface/hub")
SNAP = {
    "base": f"{HF}/models--tencent--AuK/snapshots/790742b71a4430120daf2b2099192abae449eb9f",
    "flash": f"{HF}/models--tencent--AuK-Flash/snapshots/575b92f0895f75180bf2cbd35f2e176c5732b8ed",
}
CKPT = {"base": "auk_base.safetensors", "flash": "auk_flash.safetensors"}
REPO = {"base": "AuK", "flash": "AuK-Flash"}
QWEN = f"{HF}/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e"
QWEN_SIDE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.json",
    "preprocessor_config.json",
)


def load_all(path: str) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def save(tensors: dict[str, torch.Tensor], path: str, dtype: str, keep_fp32: tuple[str, ...] = ()) -> None:
    """Save as `dtype`; names matching `keep_fp32` substrings stay float32 (norm stats, filters, fusion logits)."""
    target = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    out = {}
    for k, v in tensors.items():
        if v.is_floating_point():
            v = v.to(torch.float32 if any(s in k for s in keep_fp32) else target)
        out[k] = v.contiguous()
    save_file(out, path, metadata={"format": "mlx"})


# --------------------------------------------------------------------------- DiT


def convert_dit(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.startswith("text_encoder."):
            continue
        if k.endswith("conv_pos_embed.conv1d.0.weight") or k.endswith("conv_pos_embed.conv1d.2.weight"):
            v = v.permute(0, 2, 1).contiguous()  # (O, I/g, K) → (O, K, I/g)
        out[k] = v
    return out


# --------------------------------------------------------------------------- VAE


def fold_weight_norm(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_v"):
            base = k[: -len(".weight_v")]
            g = sd[base + ".weight_g"]
            # weight_g keeps size 1 on every dim except the norm dim; here that is dim 0 throughout
            dim = [i for i, s in enumerate(g.shape) if s != 1]
            dim = dim[0] if dim else 0
            out[base + ".weight"] = torch._weight_norm(v, g, dim)
        elif k.endswith(".weight_g"):
            continue
        else:
            out[k] = v
    return out


def convert_vae(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sd = {k: v for k, v in sd.items() if not k.startswith("flow.")}
    sd = fold_weight_norm(sd)
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight") and v.ndim == 3 and not k.endswith("filter"):
            if k.startswith("ups."):
                v = v.permute(1, 2, 0).contiguous()  # ConvTranspose1d (I, O, K) → (O, K, I)
            else:
                v = v.permute(0, 2, 1).contiguous()  # Conv1d (O, I, K) → (O, K, I)
        out[k] = v
    return out


# --------------------------------------------------------------------------- encoder


def convert_encoder() -> dict[str, torch.Tensor]:
    with open(os.path.join(QWEN, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    keep = [
        k
        for k in wm
        if (k.startswith("thinker.model.") or k.startswith("thinker.audio_tower."))
        and not k.startswith("thinker.audio_tower.audio_bos_eos_token")
    ]
    by_file: dict[str, list[str]] = {}
    for k in keep:
        by_file.setdefault(wm[k], []).append(k)
    out = {}
    for fn, ks in by_file.items():
        with safe_open(os.path.join(QWEN, fn), "pt") as f:
            for k in ks:
                v = f.get_tensor(k)
                if k in ("thinker.audio_tower.conv1.weight", "thinker.audio_tower.conv2.weight"):
                    v = v.permute(0, 2, 1).contiguous()
                out[k] = v
    return out


# --------------------------------------------------------------------------- main


def write_variant(variant: str, out_root: str, dtype: str) -> None:
    snap = SNAP[variant]
    tag = "fp32" if dtype == "float32" else "bf16"
    d = os.path.join(out_root, f"{REPO[variant]}-{tag}")
    os.makedirs(d, exist_ok=True)

    dit = convert_dit(load_all(os.path.join(snap, CKPT[variant])))
    save(dit, os.path.join(d, "dit.safetensors"), dtype, keep_fp32=("layer_weights", "layer_scale", "inv_freq"))
    vae = convert_vae(load_all(os.path.join(snap, "vae.safetensors")))
    save(vae, os.path.join(d, "vae.safetensors"), dtype, keep_fp32=("global_mean", "global_log_std", "filter"))

    with open(os.path.join(snap, "config.yaml")) as f:
        cfg = yaml.safe_load(f)["model"]
    vae_kwargs = dict(cfg["vae"]["model_init_kwargs"])
    vae_kwargs["latent_dim"] = cfg["vae"]["latent_dim"]
    arch = {k: v for k, v in cfg["arch"].items() if k not in ("attn_backend", "checkpoint_activations", "checkpoint_every_n_layers")}
    config = dict(
        model_type="auk",
        name=cfg["name"],
        is_flash=cfg["name"] == "AuK-Flash",
        dtype=dtype,
        arch=dict(arch, dim_head=64, latent_dim=cfg["vae"]["latent_dim"]),
        vae=dict(
            latent_dim=cfg["vae"]["latent_dim"],
            downsample_rate=cfg["vae"]["downsample_rate"],
            target_sample_rate=cfg["vae"]["target_sample_rate"],
            **{k: v for k, v in vae_kwargs.items() if k != "latent_dim"},
        ),
        text_encoder=dict(repo="mlx-community/Qwen2.5-Omni-3B-Thinker", num_hidden_layers=36, hidden_size=2048),
        sampling=(
            dict(nfe=4, cfg_strength=0.0, sway_sampling_coef=None, t_grid=[0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0])
            if cfg["name"] == "AuK-Flash"
            else dict(nfe=32, cfg_strength=2.0, sway_sampling_coef=-1.0, t_grid=None)
        ),
        upstream=dict(code="Tencent-Hunyuan/AuK@9481474", weights=f"tencent/{REPO[variant]}@{os.path.basename(snap)[:8]}"),
    )
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"{variant}: {len(dit)} dit + {len(vae)} vae tensors -> {d}")


def write_encoder(out_root: str, dtype: str) -> None:
    tag = "fp32" if dtype == "float32" else "bf16"
    d = os.path.join(out_root, f"Qwen2.5-Omni-3B-Thinker-{tag}")
    os.makedirs(d, exist_ok=True)
    enc = convert_encoder()
    save(enc, os.path.join(d, "encoder.safetensors"), dtype)
    with open(os.path.join(QWEN, "config.json")) as f:
        full = json.load(f)
    t = full["thinker_config"]
    config = dict(
        model_type="qwen2_5_omni_thinker_encoder",
        dtype=dtype,
        text_config={k: t["text_config"][k] for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_theta", "vocab_size", "max_position_embeddings") if k in t["text_config"]},
        audio_config={k: t["audio_config"][k] for k in ("d_model", "encoder_attention_heads", "encoder_ffn_dim", "encoder_layers", "num_mel_bins", "max_source_positions", "n_window", "output_dim", "activation_function")},
        audio_token_index=t["audio_token_index"],
        audio_start_token_id=t["audio_start_token_id"],
        audio_end_token_id=t["audio_end_token_id"],
        upstream=f"Qwen/Qwen2.5-Omni-3B@{os.path.basename(QWEN)[:8]}",
    )
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for fn in QWEN_SIDE_FILES:
        p = os.path.join(QWEN, fn)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(d, fn))
    print(f"encoder: {len(enc)} tensors -> {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--parts", default="base,flash,encoder")
    a = ap.parse_args()
    for p in a.parts.split(","):
        if p == "encoder":
            write_encoder(a.out, a.dtype)
        else:
            write_variant(p, a.out, a.dtype)


if __name__ == "__main__":
    main()
