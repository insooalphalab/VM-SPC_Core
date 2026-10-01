"""시장 차익·비차익 프로그램(베이시스 대신) — 검증이력 9.50 사전 등록 기준 그대로.

  python stock_track/collect_market_program.py   (먼저)
  python research/validate_market_arb.py         → results/market_arb_validation.json
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

import validate_signal_combo as sc
from collect_market_program import load as load_market
from v2_config import results_dir

COST = 0.003
SPLIT = pd.Timestamp("2021-06-01")


def zser(s: pd.Series) -> pd.Series:
    return (s - s.rolling(60, min_periods=40).mean().shift(1)) / s.rolling(60, min_periods=40).std().shift(1)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    mk = load_market()
    za, zn = zser(mk["arbt_net"]), zser(mk["nabt_net"])
    d = pd.read_csv(results_dir() / "winrate_events.csv", parse_dates=["entry_date"]).dropna(subset=["depth", "cpos"])
    idx = mk.index
    def min3(z, t):
        i = idx.searchsorted(t)                       # 진입일 위치 → 그 전 3거래일
        w = z.iloc[max(0, i - 3):i]
        return float(w.min()) if len(w) and w.notna().any() else np.nan
    d["arb_z"] = [min3(za, t) for t in d["entry_date"]]
    d["nabt_z"] = [min3(zn, t) for t in d["entry_date"]]
    d["win"] = ((d["ret"] - COST) > 0).astype(float)
    d["t2"] = d["t2"].astype(float)
    d["reg_side"] = (d["reg"] == "횡보·전환").astype(float)
    d["reg_down"] = (d["reg"] == "하락장").astype(float)
    d["depth"] = d["depth"].clip(0, 0.12)
    d = d.dropna(subset=["arb_z", "nabt_z"])
    base = ["t2", "reg_side", "reg_down", "depth", "cpos"]
    train, test = d[d.entry_date < SPLIT], d[d.entry_date >= SPLIT]
    res = {"n_train": int(len(train)), "n_test": int(len(test))}
    for name, col in (("차익 프로그램 z(이탈~복귀 최저)", "arb_z"), ("비차익 프로그램 z(이탈~복귀 최저)", "nabt_z")):
        r = sc.compare(train, test, base, base + [col], rng)
        r["verdict"] = "남김" if r["ci"][0] > 0 and r["logloss_new"] < r["logloss_base"] else "뺌"
        cut = d[col].quantile(0.10)
        lo_, hi_ = d[d[col] <= cut], d[d[col] > cut]
        R = lambda g: float(((g["ret"] - COST) / g["stop_pct"].clip(lower=0.01)).mean())
        r["하위10%(매도 충격) vs 나머지"] = {"cut_z": round(float(cut), 2), "승률": [round(float(lo_.win.mean()), 3), round(float(hi_.win.mean()), 3)],
                                       "R": [round(R(lo_), 3), round(R(hi_), 3)], "n": [int(len(lo_)), int(len(hi_))]}
        res[name] = r
    (results_dir() / "market_arb_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
