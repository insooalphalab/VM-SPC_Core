"""횡보·전환 장세의 박스 돌파 — 현행 돌파 규칙 + 최근경향 기준 — 검증이력 9.89 사전 등록 그대로.

  python research/validate_sideways_breakout.py   → results/sideways_breakout.json
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

import validate_box as vb
from reclassify_recent import level_ci, OLD, RECENT, HALF
from validate_breakout_exits import EXITS, simulate
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

CFG = EXITS["E2 50일선"]
END = pd.Timestamp("2100-01-01")


def run_events(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        ma = pd.Series(c).rolling(50).mean().to_numpy()
        for _, x in g.iterrows():
            e = int(b.index.searchsorted(x.entry_date))
            if e >= len(b) or b.index[e] != x.entry_date or e < 60 or np.isnan(atr[e - 1]):
                continue
            stop0 = c[e - 1] - atr[e - 1]
            sp = max(1 - stop0 / o[e], 0.01)
            ret, hold = simulate(o, h, l, c, ma, atr, e, stop0, CFG)
            rows.append({**x.to_dict(), "R2": (ret - vb.COST) / sp, "win2": float(ret - vb.COST > 0), "hold2": hold})
    return pd.DataFrame(rows)


def judge_level(x: pd.DataFrame, rng) -> dict:
    per = lambda lo, hi: x[(x.entry_date >= lo) & (x.entry_date < hi)]
    o, r = per(OLD, RECENT), per(RECENT, END)
    oc, rc = level_ci(o, rng), level_ci(r, rng)
    h1, h2 = float(per(RECENT, HALF).R.mean()), float(per(HALF, END).R.mean())
    rec_ok = rc[0] >= 0.20 and rc[1] > 0 and min(h1, h2) >= 0.10
    old_ok = oc[0] >= 0.20 and oc[1] > 0
    return {"앞 기간 R [CI]": [round(v, 3) for v in oc], "앞 건수": len(o), "앞 승률": round(float(o.win2.mean()), 3),
            "최근 R [CI]": [round(v, 3) for v in rc], "최근 건수": len(r), "최근 승률": round(float(r.win2.mean()), 3),
            "2021-06~2023": round(h1, 3), "2024~": round(h2, 3),
            "판정": ("검증됨" if old_ok else "최근경향") if rec_ok else "아님"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "base_duration_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    d = run_events(ev).assign(R=lambda x: x.R2)
    d.to_csv(results_dir() / "sideways_breakout_events.csv", index=False)
    res = {}
    for reg in ("횡보·전환", "상승장", "하락장"):
        x = d[d.reg == reg]
        groups = {"A 전체": x, "B RS ≥ 70": x[x.rs >= 0.70], "C RS ≥ 70 + 천장 나이 21~30": x[(x.rs >= 0.70) & (x.age >= 21)]}
        res[reg] = {k: judge_level(g, rng) for k, g in groups.items()}
    (results_dir() / "sideways_breakout.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for reg, v in res.items():
        for k, r in v.items():
            print(reg, k, r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
