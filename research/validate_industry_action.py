"""섹터 동반 움직임(인더스트리 액션) 돌파 — 검증이력 9.60 사전 등록 기준 그대로.

  python research/validate_industry_action.py   → results/industry_action_validation.json
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
from box_rules import BOX
from universe_all import _master, validation_codes
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
CO_WIN, CO_SHARE, CO_MIN, MIN_SECTOR = 5, 0.10, 2, 5


def sectors() -> dict[str, str]:
    out = {}
    for m, tail in (("kospi", 228), ("kosdaq", 222)):
        for ln in _master(m):
            p2, code = ln[-tail:], ln[:9].strip()
            mid, big = p2[8:12], p2[4:8]
            out[code] = f"{m}:{mid if mid.strip('0') else big}"
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    sec = sectors()
    codes = [c for c in validation_codes() if c in sec]
    brk, r60 = {}, {}
    for code in codes:
        b = load_bars(LONG_HISTORY, code)
        if b is None or len(b) < 300:
            continue
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        H = b["high"].rolling(BOX).max().shift(1)
        brk[code] = ((b["close"] > H) & (b["close"].shift(1) <= H.shift(1))).astype(float)
        r60[code] = b["close"] / b["close"].shift(60) - 1
    B = pd.DataFrame(brk).sort_index()
    R60 = pd.DataFrame(r60).sort_index()
    present = R60.notna()
    B5 = B.rolling(CO_WIN, min_periods=1).max()                    # 최근 5일 안 첫 돌파 여부(종목별)
    groups = pd.Series({c: sec[c] for c in B.columns})
    # 섹터별 매일: 종목 수, 최근 5일 돌파 종목 수, 60일 수익 평균
    n_mem = present.T.groupby(groups).sum().T
    n_brk = (B5.where(present, 0)).T.groupby(groups).sum().T
    sec_ret = R60.T.groupby(groups).mean().T.where(n_mem >= MIN_SECTOR)
    sec_rank = sec_ret.rank(axis=1, pct=True)
    ev = pd.read_csv(results_dir() / "breakout_edge_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    ev = ev[ev.code.isin(B.columns)]
    rows = []
    idx = B.index
    for _, e in ev.iterrows():
        i = idx.searchsorted(e.entry_date) - 1                     # 돌파일
        if i < 0:
            continue
        d, g = idx[i], sec[e.code]
        mem = n_mem.at[d, g] - 1
        others = n_brk.at[d, g] - B5.at[d, e.code]
        share = others / mem if mem > 0 else np.nan
        rows.append({**e.to_dict(), "sector": g, "co_share": share, "co_n": others,
                     "co": bool(mem >= MIN_SECTOR - 1 and others >= CO_MIN and share >= CO_SHARE),
                     "strong": bool(sec_rank.at[d, g] > 2 / 3) if pd.notna(sec_rank.at[d, g]) else False,
                     "ranked": bool(pd.notna(sec_rank.at[d, g]))})
    d = pd.DataFrame(rows)
    d.to_csv(results_dir() / "industry_action_events.csv", index=False)
    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "초과": round(float((g.ret - g.ctrl).mean()), 4)}
    res = {"섹터 수": int(groups.nunique()), "장세별": {}}
    for r, g in d.groupby("reg"):
        gr = g[g.ranked]
        res["장세별"][r] = {"동반 돌파": s(g[g.co]), "혼자 돌파": s(g[~g.co]), "섹터 강세(상위 1/3)": s(gr[gr.strong]),
                          "섹터 약세(나머지)": s(gr[~gr.strong]), "동반 + 강세": s(gr[gr.co & gr.strong])}
    u = d[d.reg == "상승장"]
    for name, flag in (("동반 돌파", "co"), ("섹터 강세", "strong")):
        x = u[u.ranked] if flag == "strong" else u
        a = x.assign(R=x.R)
        tr = vr.diff_ci(a[(a.entry_date < SPLIT) & a[flag]], a[(a.entry_date < SPLIT) & ~a[flag]], rng)
        te = vr.diff_ci(a[(a.entry_date >= SPLIT) & a[flag]], a[(a.entry_date >= SPLIT) & ~a[flag]], rng)
        res[f"상승장 {name} − 나머지 R"] = {"맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                       "verdict": "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
                                                  "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "industry_action_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
