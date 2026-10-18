"""V1 gate: converted tensors equal upstream's own folded weights (remove_weight_norm), layout-adjusted back."""
import os, sys
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig

HF = os.path.expanduser("~/.cache/huggingface/hub")
snap = f"{HF}/models--tencent--AuK/snapshots/790742b71a4430120daf2b2099192abae449eb9f"
W = sys.argv[1]  # converted dir, e.g. ../weights/AuK-fp32

cfg = OmegaConf.load(os.path.join(snap, "config.yaml")).model.vae
vcfg = BigVGANFlowVAEConfig.from_dict(OmegaConf.to_container(cfg.model_init_kwargs, resolve=True))
vae = load_vae_model("BigVGANFlowVAE", vcfg, os.path.join(snap, "vae.safetensors"), map_location="cpu")
for m in vae.modules():
    if hasattr(m, "weight_g"):
        torch.nn.utils.remove_weight_norm(m)
ref = {k: v for k, v in vae.state_dict().items() if not k.startswith("flow.")}

worst, n = 0.0, 0
with safe_open(os.path.join(W, "vae.safetensors"), "pt") as f:
    got = set(f.keys())
    missing = set(ref) - got
    extra = got - set(ref)
    for k in ref:
        if k not in got:
            continue
        v = f.get_tensor(k).float()
        if v.ndim == 3 and k.endswith(".weight") and not k.endswith("filter"):
            v = v.permute(2, 0, 1) if k.startswith("ups.") else v.permute(0, 2, 1)
        d = (v - ref[k].float()).abs().max().item()
        worst = max(worst, d); n += 1
print(f"VAE: {n} tensors compared, max|Δ| = {worst:.3e}, missing={sorted(missing)[:5]} ({len(missing)}), extra={sorted(extra)[:5]} ({len(extra)})")

src = {}
with safe_open(os.path.join(snap, "auk_base.safetensors"), "pt") as f:
    src = {k: f.get_tensor(k) for k in f.keys()}
worst, n = 0.0, 0
with safe_open(os.path.join(W, "dit.safetensors"), "pt") as f:
    got = set(f.keys())
    for k, r in src.items():
        v = f.get_tensor(k).float()
        if "conv_pos_embed.conv1d" in k and k.endswith("weight"):
            v = v.permute(0, 2, 1)
        worst = max(worst, (v - r.float()).abs().max().item()); n += 1
print(f"DiT: {n} tensors compared, max|Δ| = {worst:.3e}, key sets equal: {got == set(src)}")
