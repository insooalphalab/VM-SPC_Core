"""실적 성장(CAN SLIM "C"·"A") 돌파 — 검증이력 9.78 사전 등록 그대로.

  python research/validate_earnings_growth.py   → results/earnings_growth_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "stock_track", _ROOT / "dart_events"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

import validate_rebreakout as vr
from client import dart_dir
from growth import features, table
from v2_config import results_dir

SPLIT = pd.Timestamp("2021-06-01")
FLAGS = {"① 분기 순이익 +25%": "c_ni", "② 분기 매출 +25%": "c_rev", "③ 연간 순이익 +20%": "a_ni"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    fin = pd.read_csv(dart_dir() / "financials.csv", dtype={"stock_code": str, "rcept_dt": str})
    ni, rev = table(fin, "ni"), table(fin, "rev")
    ev = pd.read_csv(results_dir() / "breakout_exits_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    ev = ev[["code", "entry_date", "E2 50일선|R", "E2 50일선|win"]].rename(columns={"E2 50일선|R": "R", "E2 50일선|win": "win"})
    f = pd.DataFrame([features(c, d, ni, rev) for c, d in zip(ev.code, ev.entry_date)], index=ev.index)
    ev = ev.join(f)
    ev.to_csv(results_dir() / "earnings_growth_events.csv", index=False)
    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "맞춤/예측 R": [round(float(g.R[g.entry_date < SPLIT].mean()), 3), round(float(g.R[g.entry_date >= SPLIT].mean()), 3)]}
    res = {"n": int(len(ev)), "재무 자료 있음": round(float(ev.has.mean()), 3)}
    for name, col in FLAGS.items():
        a, b = ev[ev[col] == 1], ev[ev[col] != 1]
        tr = vr.diff_ci(a[a.entry_date < SPLIT], b[b.entry_date < SPLIT], rng)
        te = vr.diff_ci(a[a.entry_date >= SPLIT], b[b.entry_date >= SPLIT], rng)
        res[name] = {"충족": s(a), "미충족(자료 있음)": s(ev[ev[col] == 0]), "자료 없음": s(ev[ev[col].isna()]),
                     "맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                     "verdict": "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    both = ev[(ev.c_ni == 1) & (ev.c_rev == 1)]
    res["①+② 동시"] = s(both)
    res["실적 접수 20일 안 돌파"] = {"예": s(ev[ev.fresh == 1]), "아니오": s(ev[ev.fresh == 0])}
    (results_dir() / "earnings_growth_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
