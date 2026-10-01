"""박스 돌파일에 5일선 · 20일선 · 볼린저 상단 동시 돌파 — 검증이력 9.76 사전 등록 그대로.

  python research/validate_confluence.py   → results/confluence_validation.json
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
LINES = ("① 5일선", "② 20일선", "③ 볼린저 상단")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "breakout_exits_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    ev = ev[["code", "entry_date", "E2 50일선|R", "E2 50일선|win"]].rename(columns={"E2 50일선|R": "R", "E2 50일선|win": "win"})
    flags = {k: {} for k in LINES}
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        c = b["close"]
        lines = {"① 5일선": c.rolling(5).mean(), "② 20일선": c.rolling(20).mean(),
                 "③ 볼린저 상단": c.rolling(20).mean() + 2 * c.rolling(20).std()}
        for i, x in g.iterrows():
            t = int(b.index.searchsorted(x.entry_date)) - 1
            for k, ln in lines.items():
                flags[k][i] = bool(c.iloc[t] > ln.iloc[t] and c.iloc[t - 1] <= ln.iloc[t - 1])
    for k in LINES:
        ev[k] = pd.Series(flags[k])
    ev["개수"] = ev[list(LINES)].sum(1)
    ev.to_csv(results_dir() / "confluence_events.csv", index=False)
    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3)}
    res = {"n": int(len(ev)), "개수별": {int(k): s(g) for k, g in ev.groupby("개수")}}
    for k in LINES:
        a, b = ev[ev[k]], ev[~ev[k]]
        tr = vr.diff_ci(a[a.entry_date < SPLIT], b[b.entry_date < SPLIT], rng)
        te = vr.diff_ci(a[a.entry_date >= SPLIT], b[b.entry_date >= SPLIT], rng)
        res[k] = {"같은 날 돌파": s(a), "나머지": s(b), "맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                  "verdict": "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "confluence_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
