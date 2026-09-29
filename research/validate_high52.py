"""52주 신고가 돌파 리테스트 — 검증이력 9.20 사전 등록 기준 그대로(판정 = ETF 2006~2015).

  python research/validate_high52.py     → results/high52_retest_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "dart_events",
           _ROOT / "stock_track", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys

import numpy as np
import pandas as pd

from v2_config import results_dir
from v2_datastore import load_bars
from validate_box import BLOCK, LONG_BASKET
from validate_volume import diff_ci

LOOKBACK = 250
JUDGE = ("2006-01-01", "2015-12-31")


def tag(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[events["pattern"] == "retest"].copy()
    flags = {}
    for code, g in ev.groupby("code"):
        bars = load_bars(LONG_BASKET, code)
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        hi = bars["high"].rolling(LOOKBACK).max().shift(1).to_numpy()
        c = bars["close"].to_numpy(float)
        for i, b in zip(g.index, g["b"].astype(int)):
            flags[i] = np.nan if np.isnan(hi[b]) else float(c[b] > hi[b])
    ev["high52"] = pd.Series(flags)
    ev = ev.dropna(subset=["high52"])
    cal = load_bars(LONG_BASKET, "069500").index
    ev["entry_date"] = pd.to_datetime(ev["entry_date"])
    ev["ex"] = ev["ret"] - ev["ctrl_ret"]
    ev["block"] = cal.searchsorted(ev["entry_date"]) // BLOCK
    return ev


def compare(ev: pd.DataFrame, rng) -> dict:
    r = diff_ci(ev["ex"].to_numpy(), ev["high52"].astype(int).to_numpy(), ev["block"].to_numpy(), rng)
    yes, no = ev[ev["high52"] == 1], ev[ev["high52"] == 0]
    return {**r, "n_blocks": int(ev["block"].nunique()), "n_codes": int(ev["code"].nunique()),
            "target_yes": round(float((yes["how"] == "target").mean()), 4) if len(yes) else None,
            "target_no": round(float((no["how"] == "target").mean()), 4) if len(no) else None}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    etf = tag(pd.read_csv(results_dir() / "box_scenario_events_etf.csv", dtype={"code": str}))
    stk = tag(pd.read_csv(results_dir() / "box_scenario_events.csv", dtype={"code": str}))
    judge = etf[(etf["entry_date"] >= JUDGE[0]) & (etf["entry_date"] <= JUDGE[1])]
    main_r = compare(judge, rng)
    passed = main_r["diff_ci"][0] > 0 and main_r["ex_yes_ci"][0] > 0
    by_year = {}
    for name, d in (("etf", etf), ("stocks", stk)):
        yes = d[d["high52"] == 1]
        by_year[name] = {int(y): [int(len(g)), round(float(g["ex"].mean()), 4)] for y, g in yes.groupby(yes["entry_date"].dt.year)}
    out = {"verdict": "통과" if passed else "HOLD", "judge_period": JUDGE, "judge_etf_2006_2015": main_r,
           "ref_etf_2016_2026": compare(etf[etf["entry_date"] > JUDGE[1]], rng),
           "ref_stocks_2016_2026": compare(stk, rng), "high52_by_year": by_year}
    (results_dir() / "high52_retest_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
