"""V0 oracle: run upstream AuK (Tencent-Hunyuan/AuK@9481474) in fp32 on the CPU and dump goldens per stage.

Every tensor a port must match is saved as float32 .npy under goldens/<variant>/<task>/:
  processor  input_ids, attention_mask, input_features, feature_attention_mask, audio16k (what qwen_omni_utils loaded)
  thinker    audio_tower_out, hidden_{0,1,18,35,36}, fused (text_embeds), context_mask
  vae enc    ref24k (resampled input), enc_mean, enc_log_std, enc_noise, ref_latents (normalised)
  sampler    y0, t_grid, final latent (gen_latent, normalised), denorm latent fed to the decoder
  dit        call0_{x,time,text,ref,ref_mask,mask}, call0_out, call0_block{i}_{x,c} (MMDiT) / call0_single{i}
  vae dec    dec_in, audio_out
plus out.wav and meta.json (instruction, seconds, sampling knobs, formatted ChatML text).

Noise is made reproducible, not injected: the sampler seeds torch.manual_seed(seed) before randn(dur, 64), and the
VAE reference noise is drawn after torch.manual_seed(VAE_SEED). Both are saved, so a port loads them.

    .venv/bin/python dump_goldens.py --variant base|flash [--tasks zs,itts,content,speed,emotion,se]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import qwen_omni_utils
from auk.infer.infer_auk import AukInfer, save_audio


HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "ref", "AuK", "assets", "demo-input-audio")
HF = os.path.expanduser("~/.cache/huggingface/hub")
SNAP = {
    "base": f"{HF}/models--tencent--AuK/snapshots/790742b71a4430120daf2b2099192abae449eb9f",
    "flash": f"{HF}/models--tencent--AuK-Flash/snapshots/575b92f0895f75180bf2cbd35f2e176c5732b8ed",
}
CKPT = {"base": "auk_base.safetensors", "flash": "auk_flash.safetensors"}
QWEN = f"{HF}/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e"

SEED = 1234
VAE_SEED = 4321

# canonical cookbook templates (docs/COOKBOOK.md), with the cookbook's target durations
TASKS = {
    "zs": dict(
        audio="zero-shot-tts/ref.wav",
        instruction="Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the "
        "opportunity to address such a distinguished audience'",
        seconds=6.0,
    ),
    "itts": dict(
        audio=None,
        instruction='Based on the following description: "一位二十多岁的女性，面对刚到家的伴侣，以温柔、关心且略带撒娇的语气轻声诉说。'
        "她的声音柔和亲密，语速稍缓，音量适中，音色甜美自然，在短语结尾处语调温暖地上扬。"
        '她正向伴侣问候回家，并温柔询问今日工作如何。", generate speech content "Welcome home, how was work today?".',
        seconds=1.7,
    ),
    "content": dict(
        audio="content-edit/content.wav",
        instruction="Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
        seconds=7.0,
    ),
    "speed": dict(audio="speed/speed-edit-1-input.wav", instruction="Adjust the speech speed to 1.5x.", seconds=6.86),
    "emotion": dict(audio="emotion-edit/en-1-input.wav", instruction="Change the emotion to happy.", seconds=None),
    "se": dict(
        audio="se/se-zh-1-input.wav",
        instruction="Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.",
        seconds=None,
    ),
}

HIDDEN_KEEP = (0, 1, 18, 35, 36)


def npy(d: str, name: str, t) -> None:
    if isinstance(t, torch.Tensor):
        t = t.detach().to("cpu")
        t = t.float().numpy() if t.is_floating_point() else t.numpy()
    np.save(os.path.join(d, f"{name}.npy"), np.ascontiguousarray(t))


def instrument(engine: AukInfer, cap: dict) -> None:
    """Wrap the stages so each call records into `cap` (a dict reset per task)."""
    model = engine.model
    thinker = model.text_encoder

    # --- processor audio (what qwen_omni_utils hands the Whisper feature extractor) ---
    orig_mm = qwen_omni_utils.process_mm_info

    def mm(*a, **k):
        audios, images, videos = orig_mm(*a, **k)
        if audios:
            cap["audio16k"] = np.asarray(audios[0], dtype=np.float32)
        return audios, images, videos

    qwen_omni_utils.process_mm_info = mm

    # --- audio tower output ---
    def tower_hook(_m, _i, out):
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
        cap.setdefault("audio_tower_out", h.detach().clone())

    thinker.audio_tower.register_forward_hook(tower_hook)

    # --- thinker + fusion ---
    orig_encode = model.encode_text

    def encode_text(cond_inputs, device):
        for k in ("input_ids", "attention_mask", "input_features", "feature_attention_mask"):
            if k in cond_inputs:
                cap[k] = cond_inputs[k].detach().clone()
        out_hidden = {}

        def keep_hidden(_m, _i, out):
            hs = out.hidden_states
            for j in HIDDEN_KEEP:
                out_hidden[j] = hs[j].detach().clone()

        h = thinker.register_forward_hook(keep_hidden)
        try:
            hidden, mask = orig_encode(cond_inputs, device)
        finally:
            h.remove()
        for j, v in out_hidden.items():
            cap[f"hidden_{j}"] = v
        cap["fused"] = hidden.detach().clone()
        cap["context_mask"] = mask.detach().clone()
        return hidden, mask

    model.encode_text = encode_text

    # --- VAE encode with reproducible noise ---
    vae = engine.vae_model
    orig_enc = vae.encoding_and_normalization

    def enc(sample, sample_lengths=None):
        cap["ref24k"] = sample.detach().clone()
        stats = vae.audio_encoder(sample)
        mean, log_std = stats.chunk(2, 1)
        cap["enc_mean"], cap["enc_log_std"] = mean.detach().clone(), log_std.detach().clone()
        torch.manual_seed(VAE_SEED)
        cap["enc_noise"] = torch.randn_like(mean)
        torch.manual_seed(VAE_SEED)
        latents, lens = orig_enc(sample, sample_lengths)
        cap["ref_latents"] = latents.detach().clone()
        return latents, lens

    vae.encoding_and_normalization = enc

    # --- VAE decode ---
    orig_dec = vae.inference_from_latents

    def dec(x):
        cap["dec_in"] = x.detach().clone()
        y = orig_dec(x)
        cap["audio_out"] = y.detach().clone()
        return y

    vae.inference_from_latents = dec

    # --- DiT: first call inputs/outputs + per-block outputs ---
    dit = model.transformer

    def dit_pre(_m, args, kwargs):
        if "call0_out" in cap:
            return
        for k in ("x", "time", "text", "ref", "ref_mask", "mask", "c_mask"):
            v = kwargs.get(k)
            if isinstance(v, torch.Tensor):
                cap[f"call0_{k}"] = v.detach().clone()
        cap["call0_cfg"] = bool(kwargs.get("cfg_infer", False))

    def dit_post(_m, _a, _k, out):
        cap.setdefault("call0_out", out.detach().clone())

    dit.register_forward_pre_hook(dit_pre, with_kwargs=True)
    dit.register_forward_hook(dit_post, with_kwargs=True)

    for i, blk in enumerate(dit.transformer_blocks):

        def mm_hook(_m, _i, out, i=i):
            if "call0_out" not in cap and f"call0_block{i}_x" not in cap:
                c, x = out
                cap[f"call0_block{i}_c"], cap[f"call0_block{i}_x"] = c.detach().clone(), x.detach().clone()

        blk.register_forward_hook(mm_hook)
    for i, blk in enumerate(dit.single_transformer_blocks):

        def s_hook(_m, _i, out, i=i):
            if "call0_out" not in cap and f"call0_single{i}" not in cap:
                cap[f"call0_single{i}"] = out.detach().clone()

        blk.register_forward_hook(s_hook)

    # --- final latent (normalised, target slice) via denormalize ---
    orig_denorm = vae.denormalize

    def denorm(lat):
        cap["gen_latent"] = lat.detach().clone()
        return orig_denorm(lat)

    vae.denormalize = denorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("base", "flash"), required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    snap = SNAP[args.variant]
    t0 = time.time()
    engine = AukInfer(
        os.path.join(snap, "config.yaml"), os.path.join(snap, CKPT[args.variant]), device="cpu", dtype="fp32", qwen_path=QWEN
    )
    engine.model.text_encoder.to(torch.float32)  # upstream loads the thinker in bf16; the oracle is fp32 end to end
    print(f"loaded in {time.time() - t0:.1f}s")

    os.makedirs(os.path.join(HERE, "goldens", args.variant), exist_ok=True)
    npy(os.path.join(HERE, "goldens", args.variant), "layer_weights", engine.model.layer_weights)
    npy(os.path.join(HERE, "goldens", args.variant), "layer_scale", engine.model.layer_scale)

    cap: dict = {}
    instrument(engine, cap)

    for name in args.tasks.split(","):
        spec = TASKS[name]
        out_dir = os.path.join(HERE, "goldens", args.variant, name)
        os.makedirs(out_dir, exist_ok=True)
        cap.clear()

        content = [{"type": "text", "text": spec["instruction"]}]
        if spec["audio"]:
            content.append({"type": "audio", "audio": os.path.join(ASSETS, spec["audio"])})
        messages = [{"role": "user", "content": content}]

        t1 = time.time()
        audio, sr = engine.generate(messages, gen_seconds=spec["seconds"], seed=SEED)
        wall = time.time() - t1
        save_audio(audio, sr, os.path.join(out_dir, "out.wav"))

        # the sampler's y0 and time grid, recomputed exactly as CFMEdit.sample draws them
        gen_len = cap["gen_latent"].shape[1]
        torch.manual_seed(SEED)
        y0 = torch.randn(gen_len, engine.latent_dim)
        if engine.is_flash:
            tg = torch.tensor([0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0])
        else:
            tg = torch.linspace(0, 1, 33)
            tg = tg + -1.0 * (torch.cos(torch.pi / 2 * tg) - 1 + tg)
        cap["y0"], cap["t_grid"] = y0, tg

        for k, v in cap.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                npy(out_dir, k, v)

        formatted = engine.model.text_processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        meta = dict(
            variant=args.variant,
            task=name,
            instruction=messages[0]["content"][0]["text"],  # includes |<no_prompt_audio>| when appended
            audio=spec["audio"],
            gen_seconds=spec["seconds"],
            gen_latent_len=gen_len,
            seed=SEED,
            vae_seed=VAE_SEED,
            cfg=0.0 if engine.is_flash else 2.0,
            nfe=4 if engine.is_flash else 32,
            call0_cfg=cap.get("call0_cfg"),
            chatml=formatted,
            wall_s=round(wall, 1),
            torch=torch.__version__,
            upstream="Tencent-Hunyuan/AuK@9481474",
        )
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[{args.variant}/{name}] {wall:.1f}s  gen_latent={gen_len}  -> {out_dir}")


if __name__ == "__main__":
    main()
