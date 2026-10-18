"""V2b scorer (PORTING-SPEC §6). For every render under <v2b>/<tier>/<variant>/renders.jsonl:
  error rate  WER (en) / CER (zh), mlx-whisper large-v3-turbo at temperature 0, against the oracle's own transcript
  sim_src     CAM++ (funasr campplus_cn_common) output vs the task's source / reference clip
  dnsmos      DNSMOS OVRL
Normalisers, CAM++ front end and DNSMOS call follow mlxengine-audio/WIP/speech-edit-eval/harness/score.py.

    AUK_FUNASR_DETACH_ROOT=<Step-Audio-EditX clone> python score_v2b.py <v2b-dir> <goldens-dir> <assets-dir> <out.csv>
needs: mlx-whisper, jiwer, openai-whisper (normalisers), zhconv, speechmos, onnxruntime, funasr, pandas, librosa, torch
"""

import functools
import glob
import json
import os
import re
import sys

import jiwer
import librosa
import mlx_whisper
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from zhconv import convert as zh_simp

# CAM++ comes from Step-Audio-EditX's vendored funasr_detach (a clone of stepfun-ai/Step-Audio-EditX); point
# AUK_FUNASR_DETACH_ROOT at that clone. The default is the fleet's checkout.
EDITX_REF = os.environ.get("AUK_FUNASR_DETACH_ROOT",
                           "/Volumes/Satechi/Development/mlxengine-audio/WIP/speech-edit-eval/ref/Step-Audio-EditX")
sys.path.insert(0, EDITX_REF)
from funasr_detach.models.campplus.model import CAMPPlus  # noqa: E402

WHISPER = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/*/"))[0]
CAMPPLUS = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--funasr--campplus/snapshots/*/campplus_cn_common.bin"))[0]

LANG = {"zs": "en", "itts": "en", "content": "en", "speed": "en", "emotion": "en", "se": "zh"}

_en, _basic = EnglishTextNormalizer(), BasicTextNormalizer()
_zh_punct = re.compile(r"[\s，。！？、；：“”‘’（）《》〈〉【】…—\-,.!?;:'\"()\[\]]+")


def normalize(text, lang):
    if lang == "en":
        return _en(text)
    t = _zh_punct.sub("", _basic(text))
    return zh_simp(t, "zh-cn")


@functools.lru_cache(maxsize=None)
def transcribe(path, lang):
    r = mlx_whisper.transcribe(path, path_or_hf_repo=WHISPER, language=lang, temperature=0.0, condition_on_previous_text=False)
    return r["text"].strip()


def error_rate(ref, hyp, lang):
    r, h = normalize(ref, lang), normalize(hyp, lang)
    if not r:
        return float("nan")
    return jiwer.wer(r, h) if lang == "en" else jiwer.cer(r, h)


def load_mono(path, sr):
    d, s = sf.read(path, dtype="float32", always_2d=True)
    d = d.mean(axis=1)
    if s != sr:
        d = librosa.resample(d, orig_sr=s, target_sr=sr)
    return d


_camp = CAMPPlus(feat_dim=80, embedding_size=192)
_camp.load_state_dict(torch.load(CAMPPLUS, map_location="cpu"))
_camp.eval()


@functools.lru_cache(maxsize=None)
def spk_embed(path):
    d = load_mono(path, 16000)
    feat = kaldi.fbank(torch.from_numpy(d)[None], num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        e = _camp(feat[None])
    return torch.nn.functional.normalize(e, dim=-1)[0].numpy()


def sim(a, b):
    return float(np.dot(spk_embed(a), spk_embed(b)))


@functools.lru_cache(maxsize=None)
def dnsmos(path):
    from speechmos import dnsmos as dm

    d = np.clip(load_mono(path, 16000), -1.0, 1.0)
    r = dm.run(d, sr=16000, return_df=False)
    r = r[0] if isinstance(r, list) else r
    return float(r["ovrl_mos"])


def main():
    v2b, goldens, assets, out = sys.argv[1:5]
    ref_text = {}
    for variant in ("base", "flash"):
        for task in LANG:
            ref_text[(variant, task)] = transcribe(os.path.join(goldens, variant, task, "out.wav"), LANG[task])
    rows = []
    for log in sorted(glob.glob(os.path.join(v2b, "*", "*", "renders.jsonl"))):
        tier, variant = log.split(os.sep)[-3], log.split(os.sep)[-2]
        for line in open(log):
            r = json.loads(line)
            task = r["task"]
            meta = json.load(open(os.path.join(goldens, variant, task, "meta.json")))
            hyp = transcribe(r["path"], LANG[task])
            rec = dict(tier=tier, variant=variant, task=task, seed=r["seed"], hyp=hyp,
                       err=error_rate(ref_text[(variant, task)], hyp, LANG[task]),
                       sim_src=sim(r["path"], os.path.join(assets, meta["audio"])) if meta.get("audio") else float("nan"),
                       dnsmos=dnsmos(r["path"]), wall_s=r["wall_s"], audio_s=r["audio_s"], peak_gb=r["peak_gb"])
            rows.append(rec)
            print(f"{tier} {variant}/{task} s{r['seed']}: err {rec['err']:.3f} sim {rec['sim_src']:.3f} dnsmos {rec['dnsmos']:.2f} | {hyp}")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    # the pre-registered pass rule against tier R
    summ = df.groupby("tier")[["err", "sim_src", "dnsmos"]].mean()
    print("\n== pooled means\n", summ.round(4))
    per = df.groupby(["tier", "variant", "task"])["err"].mean().unstack("tier")
    print("\n== error rate per (variant, task)\n", per.round(3))
    if "R" in summ.index:
        r = summ.loc["R"]
        for t in summ.index:
            if t == "R":
                continue
            d_err = (summ.loc[t, "err"] - r["err"]) * 100
            worst = ((per[t] - per["R"]) * 100).max() if t in per else float("nan")
            d_sim = summ.loc[t, "sim_src"] - r["sim_src"]
            d_dns = summ.loc[t, "dnsmos"] - r["dnsmos"]
            ok = d_err <= 2.0 and worst <= 10.0 and d_sim >= -0.03 and d_dns >= -0.15
            print(f"tier {t}: Δerr {d_err:+.2f} pts (worst cell {worst:+.1f}), Δsim {d_sim:+.4f}, Δdnsmos {d_dns:+.3f} → {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
