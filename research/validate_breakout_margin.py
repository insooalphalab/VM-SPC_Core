"""돌파 마진 상한("얌전하게 뚫은 종목만") — 검증이력 9.67 사전 등록 그대로.

  python research/validate_breakout_margin.py   → results/breakout_margin_validation.json
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
from v2_config import results_dir

SPLIT = pd.Timestamp("2021-06-01")
CALM, HOT, STOP_R = 0.02, 0.04, -0.95


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "trend_principles_events.csv", parse_dates=["entry_date"])
    ev["band"] = np.where(ev.strength <= CALM, "얌전(≤2%)", np.where(ev.strength > HOT, "과열(>4%)", "중간(2~4%)"))
    ev["stopped"] = ev.R <= STOP_R
    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "평균 수익": round(float(g.ret.mean()), 4), "손절 거리(중앙)": round(float(g.stop_pct.median()), 4),
                   "손절 종료": round(float(g.stopped.mean()), 3)}
    res = {"장세별": {r: {b: s(x) for b, x in g.groupby("band")} for r, g in ev.groupby("reg")}}
    u = ev[ev.reg == "상승장"]
    for scope, x in (("전체", u), ("RS ≥ 70 안", u[u.rs >= 0.70])):
        a, b = x[x.band == "얌전(≤2%)"], x[x.band == "과열(>4%)"]
        tr = vr.diff_ci(a[a.entry_date < SPLIT], b[b.entry_date < SPLIT], rng)
        te = vr.diff_ci(a[a.entry_date >= SPLIT], b[b.entry_date >= SPLIT], rng)
        res[f"상승장 {scope} 얌전 − 과열 R"] = {"n": [int(len(a)), int(len(b))], "맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                            "verdict": "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
                                                       "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "breakout_margin_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
