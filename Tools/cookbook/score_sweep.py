"""Score sweep_phrasing.sh output with the cookbook verifier's own checks (AB-T-0416).

    python -I score_sweep.py <assets-dir> <sweep-dir>        (<sweep-dir>/<variant>/<case>-<pkg|cb|zh>-s<seed>.wav)
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_cookbook import check, load, tilt  # noqa: E402


def score(case: str, path: str, A: str) -> tuple[bool, str]:
    if case == "4.5en":  # to-whisper on an EN take: the same tilt test against that take
        ts, td = tilt(load(f"{A}/emotion-edit/en-1-input.wav")), tilt(load(path))
        return td > ts, f"HF/LF ratio {ts:.3f} -> {td:.3f} (whisper raises it)"
    cid = {"1.2": "1.2-instruct-tts", "4.5": "4.5-whisper", "fw": "S-from-whisper"}[case]
    return check(cid, path, A)


A, root = sys.argv[1], sys.argv[2]
for vdir in sorted(glob.glob(os.path.join(root, "*/"))):
    v = os.path.basename(vdir.rstrip("/"))
    tally: dict[str, list[bool]] = {}
    for p in sorted(glob.glob(os.path.join(vdir, "*.wav"))):
        case, phr, seed = re.match(r"([\w.]+?)-(pkg|cb|zh)-s(\d+)\.wav", os.path.basename(p)).groups()
        ok, detail = score(case, p, A)
        if case == "fw":  # whisper→normal: the verifier only reports it; here, the HF/LF direction (the source is not whispered)
            ok = float(detail.split("-> ")[1].split()[0]) < float(detail.split("ratio ")[1].split()[0])
        tally.setdefault(f"{case} {phr}", []).append(ok)
        print(f"  {v:5s} {case:5s} {phr:3s} seed {seed}  {'PASS' if ok else 'FAIL'}  {detail}")
    for k, oks in tally.items():
        print(f"{v:5s} {k:9s} {sum(oks)}/{len(oks)} seeds pass")
