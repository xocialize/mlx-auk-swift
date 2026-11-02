"""Cookbook verifier for MLXAuK (AB-T-0416): did each docs/COOKBOOK.md task do what it claims?

    python -I verify_cookbook.py <assets-dir> <render-dir> [<render-dir> ...] [--json out.json]

Each <render-dir> is what `auk-gates cookbook` wrote: <id>.wav per case plus cases.jsonl. The checks are ported
from upstream's scripts/verify_cookbook_mlx.py (Tencent-Hunyuan/AuK, branch feat/mlx-apple-silicon @ 6943a1e;
MIT, Copyright (C) 2026 Tencent — notice in LICENSE-upstream-AuK next to this file), with the same thresholds:
content edits are transcribed, pitch is measured in cents, volume in dB, speed by the duration hit, enhancement by
the noise floor, whisper by the spectral tilt, separation by the drop in active frames.

Differences from upstream, all in the measuring, none in the thresholds:
  * ASR is mlx-whisper large-v3-turbo from the local HF cache (upstream: whisper-small[.en]), temperature 0;
  * audio is read with scipy and resampled with resample_poly (upstream: soundfile + soxr VHQ);
  * `supplementary` rows (declared edit kinds the cookbook has no example for) get their own checks and are
    reported apart from the cookbook count;
  * 5.4 also checks WHICH speaker was kept: ss/en-1 is a two-person dialogue whose lines are known (A asks, B
    answers), so the kept stem must carry B's lines and none of A's. The same check judges the supplementary
    first- and second-speaker rows. 5.3 also checks that the lyric survives.
"""

from __future__ import annotations

import difflib
import functools
import glob
import json
import math
import os
import re
import sys
from fractions import Fraction

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as ss

SR = 24000
WHISPER = (glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/*/"))
           or ["mlx-community/whisper-large-v3-turbo"])[0]


def resample(x: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return x
    f = Fraction(to, sr).limit_denominator(1000)
    return ss.resample_poly(x, f.numerator, f.denominator).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _read(path: str) -> tuple[np.ndarray, int]:
    sr, w = wavfile.read(path)
    w = w.astype(np.float32) / (32768.0 if w.dtype == np.int16 else 2147483648.0 if w.dtype == np.int32 else 1.0)
    if w.ndim == 2:
        w = w.mean(axis=1)
    return w, sr


def load(path: str) -> np.ndarray:
    w, sr = _read(path)
    return resample(w, sr, SR)


def median_f0(x: np.ndarray) -> float:
    """Autocorrelation pitch, median over voiced frames."""
    fl, hop = int(0.04 * SR), int(0.02 * SR)
    f0 = []
    for i in range(0, max(0, len(x) - fl), hop):
        fr = x[i : i + fl]
        if np.sqrt((fr**2).mean()) < 0.02:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[fl - 1 :]
        lo, hi = int(SR / 400), int(SR / 70)
        if hi >= len(ac):
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.3 * ac[0]:
            f0.append(SR / k)
    return float(np.median(f0)) if f0 else 0.0


def cents(f0s: float, f0d: float) -> float:
    return 1200 * math.log2(f0d / f0s) if f0s > 0 and f0d > 0 else 0.0


def db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt((x**2).mean()) + 1e-12)


def frame_rms(x: np.ndarray) -> np.ndarray:
    fl, hop = int(0.025 * SR), int(0.010 * SR)
    return np.array([np.sqrt((x[i : i + fl] ** 2).mean()) for i in range(0, max(0, len(x) - fl), hop)])


def speech_rate(x: np.ndarray) -> float:
    """Syllable-ish rate: energy-envelope peaks per second."""
    e = frame_rms(x)
    if len(e) < 3:
        return 0.0
    e = e / (e.max() + 1e-9)
    peaks = ((e[1:-1] > e[:-2]) & (e[1:-1] > e[2:]) & (e[1:-1] > 0.25)).sum()
    return float(peaks / (len(e) * 0.010))


def noise_floor(x: np.ndarray) -> float:
    return float(np.percentile(frame_rms(x), 10))


def tilt(x: np.ndarray) -> float:
    """Spectral tilt: whisper pushes energy up in frequency."""
    f, p = ss.welch(x, SR, nperseg=1024)
    return float(p[(f >= 2000) & (f < 8000)].sum() / (p[(f >= 0) & (f < 1000)].sum() + 1e-12))


def active(x: np.ndarray) -> float:
    return float((frame_rms(x) > 0.02).mean())


def low_share(x: np.ndarray) -> float:
    f, p = ss.welch(x, SR, nperseg=2048)
    return float(p[f < 200].sum() / (p.sum() + 1e-12))


@functools.lru_cache(maxsize=None)
def asr(path: str, lang: str = "en") -> str:
    import mlx_whisper

    w, sr = _read(path)
    audio = resample(w, sr, 16000)
    r = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER, language=lang, temperature=0.0,
                               condition_on_previous_text=False)
    return r["text"].strip()


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z' ]", "", s.lower()).split()


def word_acc(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, words(a), words(b)).ratio()


# ss/en-1-input.wav (whisper large-v3-turbo segments): speaker A opens and asks, speaker B answers.
# "Sophie" is left out: both of them may say it.
EN1_A = ["where did you get that", "what's her name", "are you from sydney", "where do you study", "so what major"]
EN1_B = ["get what", "yeah student", "on break", "banking and finance"]


def speaker_lines(o: str) -> tuple[int, int, str]:
    """(A lines heard, B lines heard, transcript) for a stem kept from ss/en-1."""
    got = " ".join(words(asr(o)))
    hit = lambda lines: sum(" ".join(words(line)) in got for line in lines)
    return hit(EN1_A), hit(EN1_B), got


# --- per-case checks: (passed, detail). `o` is the rendered wav, `A` the assets directory. -------------------------

def check(cid: str, o: str, A: str) -> tuple[bool | None, str]:
    """(passed, detail); passed is None for a row reported but not judged."""
    if cid == "1.1-zeroshot-tts":
        target = "ladies and gentlemen it's an honor to have the opportunity to address such a distinguished audience"
        got = asr(o)
        acc = word_acc(target, got)
        return acc > 0.8, f"word-acc={acc * 100:.0f}%  “{got[:70]}”"
    if cid == "1.2-instruct-tts":
        got = asr(o)
        return "welcome home" in got.lower(), f"“{got[:70]}”"
    if cid == "2.1-content-edit":
        got = asr(o).lower()
        new_in = "dreams unmet" in got or "living well" in got
        old_out = "accepting what we cannot have" not in got
        return new_in and old_out, f"new={new_in} old_gone={old_out}  “{got[:70]}”"
    if cid == "2.2-lyric-edit":
        got = asr(o).lower()
        return "like you" in got, f"has_like_you={'like you' in got} has_rear_view={'rear view' in got}  “{got[:60]}”"
    if cid == "3.1-pitch":
        src, dst = load(f"{A}/pitch/pitch-1-input.wav"), load(o)
        f0s, f0d = median_f0(src), median_f0(dst)
        c = cents(f0s, f0d)
        return 80 < c < 330, f"{f0s:.0f}Hz -> {f0d:.0f}Hz = {c:+.0f} cents (target +200)"
    if cid == "3.2-speed":
        # the model is told the target LENGTH, so check the duration it hit (syllable rate reported, not asserted)
        src, dst = load(f"{A}/speed/speed-edit-1-input.wav"), load(o)
        rs, rd = speech_rate(src), speech_rate(dst)
        ds, dd = len(src) / SR, len(dst) / SR
        r = ds / dd if dd > 0 else 0
        return 1.35 < r < 1.7, f"dur {ds:.1f}s -> {dd:.1f}s = {r:.2f}x (target 1.5x); syllable rate {rs:.2f} -> {rd:.2f}/s"
    if cid == "3.3-volume":
        src, dst = load(f"{A}/energy/energy-edit-1-input.wav"), load(o)
        d = db(dst) - db(src)
        return d > 3, f"{db(src):.1f} -> {db(dst):.1f} dB = {d:+.1f} dB (target +10)"
    if cid == "4.1-emotion":
        srcp = f"{A}/emotion-edit/en-1-input.wav"
        src, dst = load(srcp), load(o)
        keep = word_acc(asr(srcp), asr(o))
        f0s, f0d = median_f0(src), median_f0(dst)
        return keep > 0.5, f"content kept={keep * 100:.0f}%  F0 {f0s:.0f}->{f0d:.0f}Hz ({abs(cents(f0s, f0d)):+.0f} cents)"
    if cid == "4.2-timbre":
        src, dst = load(f"{A}/vc/vc-1-input.wav"), load(o)
        f0s, f0d = median_f0(src), median_f0(dst)
        c = cents(f0s, f0d)
        return c < 0, f"F0 {f0s:.0f} -> {f0d:.0f}Hz = {c:+.0f} cents (want lower)"
    if cid == "4.3-deaccent":
        src, dst = load(f"{A}/accent/accent-sichuan-input.wav"), load(o)
        got = asr(o, "zh")
        ratio = len(dst) / len(src)
        return 0.9 < ratio < 1.1 and bool(got), f"len ratio={ratio:.2f}  “{got[:40]}”"
    if cid == "4.4-nonverbal-remove":
        src, dst = load(f"{A}/nv/en-d-input.wav"), load(o)
        return len(dst) > 0, (f"{len(src) / SR:.1f}s -> {len(dst) / SR:.1f}s  rms {np.sqrt((src**2).mean()):.4f} -> "
                              f"{np.sqrt((dst**2).mean()):.4f}")
    if cid == "4.4-nonverbal-add":
        src, dst = load(f"{A}/nv/en-c-input.wav"), load(o)
        want, got = 10.44, len(dst) / SR
        return abs(got - want) < 0.3, f"{len(src) / SR:.1f}s -> {got:.1f}s (cookbook asked for {want}s)"
    if cid == "4.5-whisper":
        src, dst = load(f"{A}/whisper/wh-w2n-zh-input.wav"), load(o)
        ts, td = tilt(src), tilt(dst)
        return td > ts, f"HF/LF ratio {ts:.3f} -> {td:.3f} (whisper raises it)"
    if cid == "5.1-enhance":
        src, dst = load(f"{A}/se/se-zh-1-input.wav"), load(o)
        ns, nd = noise_floor(src), noise_floor(dst)
        return nd < ns, f"noise floor {ns:.5f} -> {nd:.5f} ({20 * np.log10((nd + 1e-12) / (ns + 1e-12)):+.1f} dB)"
    if cid == "5.2-separation":
        asrc, adst = active(load(f"{A}/ss/zh-1-input.wav")), active(load(o))
        return adst < asrc, f"voiced-frame share {asrc:.2f} -> {adst:.2f} (fewer speakers kept)"
    if cid in ("5.4-tse", "S-speaker1-en", "S-speaker2-en"):
        asrc, adst = active(load(f"{A}/ss/en-1-input.wav")), active(load(o))
        a, b, got = speaker_lines(o)
        want_b = cid != "S-speaker1-en"
        right = (b >= 2 and a == 0) if want_b else (a >= 2 and b == 0)
        return adst < asrc and right, (f"voiced {asrc:.2f} -> {adst:.2f}; lines heard A {a}/{len(EN1_A)}, B {b}/{len(EN1_B)} "
                                       f"(want {'B' if want_b else 'A'})  “{got[:60]}”")
    if cid in ("5.3-music-sep", "S-voices"):
        ls, ld = low_share(load(f"{A}/vocal-extraction/vocal-1-input.wav")), low_share(load(o))
        lyric = "time stands still" in " ".join(words(asr(o)))
        return ld < ls * 0.5 and lyric, f"<200Hz share {ls:.3f} -> {ld:.3f} (backing removed); lyric kept={lyric}"
    # --- supplementary: declared kinds with no cookbook example ---
    if cid == "S-insert":
        got = asr(o).lower()
        ins, kept = "quietly" in got, "accepting" in got
        return ins and kept, f"inserted={ins} anchor_kept={kept}  {len(load(o)) / SR:.1f}s  “{got[:70]}”"
    if cid == "S-remove":
        srcp = f"{A}/content-edit/content.wav"
        got = asr(o).lower()
        gone = "accepting what we cannot have" not in got
        ds, dd = len(load(srcp)) / SR, len(load(o)) / SR
        return gone and dd < ds, f"removed={gone} {ds:.1f}s -> {dd:.1f}s  “{got[:70]}”"
    if cid == "S-nonverbal-add-start":
        srcp = f"{A}/nv/en-c-input.wav"
        keep = word_acc(asr(srcp), asr(o))
        want, got = 14.3, len(load(o)) / SR
        return keep > 0.5 and abs(got - want) < 0.3, f"content kept={keep * 100:.0f}%  {got:.1f}s (asked {want}s)"
    if cid == "S-dereverb":
        src, dst = load(f"{A}/se/se-zh-1-input.wav"), load(o)
        ratio = len(dst) / len(src)
        got = asr(o, "zh")
        return 0.9 < ratio < 1.1 and bool(got), (f"len ratio={ratio:.2f}  noise floor {noise_floor(src):.5f} -> "
                                                f"{noise_floor(dst):.5f}  “{got[:30]}”")
    if cid == "S-from-whisper":
        # reported, not judged: the assets' one whisper sample tilts like plain speech (HF/LF 0.073; the EN take 0.057),
        # so there is no whispered input to bring back to a normal voice
        src, dst = load(f"{A}/whisper/wh-w2n-zh-input.wav"), load(o)
        return None, f"HF/LF ratio {tilt(src):.3f} -> {tilt(dst):.3f} (information only: no whispered input in the assets)"
    raise KeyError(cid)


def verify(A: str, out_dir: str) -> list[dict]:
    rows = [json.loads(line) for line in open(os.path.join(out_dir, "cases.jsonl")) if line.strip()]
    for r in rows:
        wav = os.path.join(out_dir, f"{r['id']}.wav")
        if r["kind"] in ("cookbook", "supplementary"):
            if r.get("status") != "ok" or not os.path.isfile(wav):
                r["verdict"], r["detail"] = "FAIL", f"render failed: {r.get('error', 'no wav')[:120]}"
            else:
                ok, detail = check(r["id"], wav, A)
                r["verdict"], r["detail"] = "info" if ok is None else "PASS" if ok else "FAIL", detail
        else:
            r["verdict"], r["detail"] = "n/a", r["note"]
    return rows


def main() -> None:
    argv = sys.argv[1:]
    out_json = None
    if "--json" in argv:
        i = argv.index("--json"); out_json = argv[i + 1]; del argv[i : i + 2]
    A, dirs = argv[0], argv[1:]
    print(f"# ASR: {WHISPER}")
    allrows = {}
    for d in dirs:
        rows = verify(A, d)
        label = f"{rows[0]['variant']} {rows[0]['quant']} seed {rows[0]['seed']}"
        allrows[label] = rows
        print(f"\n## {label} — {d}")
        for r in rows:
            print(f"  {r['verdict']:4s} {r['kind']:13s} {r['id']:22s} {r['detail']}")
        cb = [r for r in rows if r["kind"] == "cookbook"]
        sup = [r for r in rows if r["kind"] == "supplementary" and r["verdict"] != "info"]
        # supplementary rows are named S-…; an unexpressible one is not a cookbook example left out
        na = [r for r in rows if r["kind"] in ("unexpressible", "excluded") and not r["id"].startswith("S-")]
        print(f"\n  cookbook: {sum(r['verdict'] == 'PASS' for r in cb)}/{len(cb)} expressible examples pass "
              f"({len(na)} of 17 not run: {', '.join(r['id'] for r in na)}); "
              f"supplementary: {sum(r['verdict'] == 'PASS' for r in sup)}/{len(sup)}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(allrows, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
