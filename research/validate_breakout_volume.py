"""돌파일 거래량 배수(직전 20일 평균 대비) — 검증이력 9.77 사전 등록 그대로.

  python research/validate_breakout_volume.py   → results/breakout_volume_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

import validate_rebreakout as vr
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
BINS, LAB = [0, 1, 1.5, 2, 3, 5, np.inf], ["< 1", "1~1.5", "1.5~2", "2~3", "3~5", "≥ 5"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "breakout_exits_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    ev = ev[["code", "entry_date", "E2 50일선|R", "E2 50일선|win"]].rename(columns={"E2 50일선|R": "R", "E2 50일선|win": "win"})
    vx = {}
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        v = b["volume"].to_numpy(float)
        avg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        for i, x in g.iterrows():
            t = int(b.index.searchsorted(x.entry_date)) - 1
            vx[i] = v[t] / avg[t] if avg[t] > 0 else np.nan
    ev["vx"] = pd.Series(vx)
    ev = ev.dropna(subset=["vx"])
    ev["band"] = pd.cut(ev.vx, BINS, labels=LAB, right=False).astype(str)
    ev.to_csv(results_dir() / "breakout_volume_events.csv", index=False)
    fit, test = ev.entry_date < SPLIT, ev.entry_date >= SPLIT
    s = lambda g: {"n": int(len(g)), "비중": round(len(g) / len(ev), 3), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "맞춤/예측 R": [round(float(g.R[fit[g.index]].mean()), 3), round(float(g.R[test[g.index]].mean()), 3)]}
    res = {"n": int(len(ev)), "배수 중앙": round(float(ev.vx.median()), 2), "구간별": {k: s(ev[ev.band == k]) for k in LAB}}
    a, b = ev[ev.vx >= 2], ev[ev.vx < 1.5]
    tr = vr.diff_ci(a[a.entry_date < SPLIT], b[b.entry_date < SPLIT], rng)
    te = vr.diff_ci(a[a.entry_date >= SPLIT], b[b.entry_date >= SPLIT], rng)
    v = ("채택(많을수록 좋음)" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
         "채택(적을수록 좋음)" if te[0] <= -0.15 and te[2] < 0 and tr[0] < 0 else
         "방향만 일치" if (te[0] > 0 and tr[0] > 0) or (te[0] < 0 and tr[0] < 0) else "채택 안 함")
    res["많음(≥2) − 적음(<1.5) R"] = {"n": [int(len(a)), int(len(b))], "맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te], "verdict": v}
    (results_dir() / "breakout_volume_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
