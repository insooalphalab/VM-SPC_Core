"""돌파일 프로그램 순매도면 돌파가 약한가 — 장세 3분할 · 돌파 전체 — 검증이력 9.100 사전 등록 그대로.

  python research/validate_breakout_prog_sell.py   → results/breakout_prog_sell.json  (먼저 validate_sideways_breakout.py)
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
from collect_investor_detail import load_detail
from collect_program import load_program
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT, HALF, END = pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
T = 0.15


def attach(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in d.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        prog = load_program(code)
        if b is None or prog is None or prog.empty:
            continue
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        val = prog["value"].reindex(b.index).replace(0, np.nan)
        f = (prog["prog_net"].reindex(b.index) / val).to_numpy()
        v = b["volume"].to_numpy(float)
        va = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        inv = load_detail(code)
        fi = ((inv.reindex(b.index)["외국인"] + inv.reindex(b.index)["기관합계"]) * 1e6 / val).to_numpy() if inv is not None and not inv.empty else None
        pos = b.index.searchsorted(g.entry_date) - 1
        g = g.copy()
        g["f"] = [f[p] if p >= 0 else np.nan for p in pos]
        g["vx3"] = [bool(v[p] >= 3 * va[p]) if p >= 20 else False for p in pos]
        g["fi"] = [fi[p] if fi is not None and p >= 0 else np.nan for p in pos]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def cell(x: pd.DataFrame, col: str, rng) -> dict:
    a, b = x[x[col] < 0], x[x[col] > 0]
    per = lambda z, lo, hi: z[(z.entry_date >= lo) & (z.entry_date < hi)]
    out = {"n 순매도/순매수": [len(a), len(b)], "R 순매도/순매수": [round(float(a.R.mean()), 3), round(float(b.R.mean()), 3)]}
    for pn, lo, hi in (("앞", pd.Timestamp("2000-01-01"), SPLIT), ("최근", SPLIT, END)):
        aa, bb = per(a, lo, hi), per(b, lo, hi)
        out[pn] = [round(v, 3) for v in vr.diff_ci(aa, bb, rng)] if len(aa) >= 20 and len(bb) >= 20 else None
    out["반쪽"] = [round(float(per(a, SPLIT, HALF).R.mean() - per(b, SPLIT, HALF).R.mean()), 3),
                  round(float(per(a, HALF, END).R.mean() - per(b, HALF, END).R.mean()), 3)]
    return out


def verdict(c: dict) -> str:
    o, r, h = c["앞"], c["최근"], c["반쪽"]
    if not o or not r:
        return "표본 부족"
    excl = lambda d: d[1] > 0 or d[2] < 0
    if r[0] <= -T and o[0] <= -T and excl(o) and excl(r):
        return "검증됨"
    if r[0] <= -T and excl(r) and h[0] < 0 and h[1] < 0 and min(abs(h[0]), abs(h[1])) >= T / 2:
        return "최근경향"
    if o[0] < 0 and r[0] < 0 and h[0] < 0 and h[1] < 0:
        return "약함(네 구간 같은 방향)"
    return "아님"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "sideways_breakout_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    d = d.assign(code=d.code.str.zfill(6), R=d.R2)[["code", "entry_date", "reg", "rs", "R"]]
    d = attach(d).dropna(subset=["f"])
    d.to_csv(results_dir() / "breakout_prog_sell_events.csv", index=False)
    res = {}
    main_c = cell(d[(d.reg == "상승장") & (d.rs >= 0.70)], "f", rng)
    main_c["판정"] = verdict(main_c)
    res["주 판정: 상승장 · RS ≥ 70"] = main_c
    for rg in ("상승장", "횡보·전환", "하락장"):
        c = cell(d[d.reg == rg], "f", rng)
        c["판정(서술)"] = verdict(c)
        res[f"서술: {rg} 전체"] = c
    up = d[(d.reg == "상승장") & (d.rs >= 0.70)].copy()
    q = up.groupby(up.entry_date.dt.year).f.rank(pct=True)
    up["fq"] = np.where(q <= 1 / 3, -1.0, np.where(q > 2 / 3, 1.0, np.nan))
    res["서술: 상승장 RS≥70 하위 1/3 − 상위 1/3"] = cell(up.dropna(subset=["fq"]), "fq", rng)
    res["서술: 상승장 RS≥70 거래량 3배 돌파만"] = cell(up[up.vx3], "f", rng)
    iv = up.dropna(subset=["fi"])
    res["서술: 투자자 판(외국인+기관) 상승장 RS≥70"] = cell(iv, "fi", rng)
    (results_dir() / "breakout_prog_sell.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
