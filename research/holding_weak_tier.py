"""보유 판단 '약함' 등급 — 장세별, 겹치지 않는 요소만 — 검증이력 9.87 사전 등록 그대로.

  python research/holding_weak_tier.py   → results/holding_weak_tier.json
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

from validate_compression_regime import regime
from validate_holding_factors import FACTORS, spreads
from validate_sideways_holding import flow_cols
from v2_config import results_dir

RECENT, HALF, T, RHO = pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), 0.005, 0.5
ANCHOR = {"상승장": "trend", "하락장": "trend", "횡보·전환": "ma60"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    pnl = pd.read_csv(results_dir() / "holding_factors_panel.csv", parse_dates=["date"]).set_index("date")
    pnl["trend"] = pnl[["ma60", "ma120", "hi52", "rs"]].groupby(level=0).rank(pct=True).mean(axis=1, skipna=False)
    fl = flow_cols(pnl)[["code", "prog20w", "prog60w", "prog120w"]]
    pnl = pnl.reset_index().merge(fl.reset_index().rename(columns={"index": "date"}), on=["date", "code"], how="left").set_index("date")
    pnl["reg"] = regime().reindex(pnl.index).to_numpy()
    cands = {**FACTORS, "프로그램 60일": "prog60w", "프로그램 120일": "prog120w"}
    res = {}
    for reg, anchor in ANCHOR.items():
        x = pnl[(pnl.reg == reg) & (pnl.index >= RECENT)]
        rk = x[list(cands.values()) + ["trend"]].groupby(level=0).rank(pct=True)
        rows = []
        for name, col in cands.items():
            if col == anchor:
                continue
            try:
                sp = spreads(x, col, "x20").spread
            except KeyError:
                continue
            m, h1, h2 = float(sp.mean()), float(sp[sp.index < HALF].mean()), float(sp[sp.index >= HALF].mean())
            rho = float(rk[col].corr(rk[anchor]))
            weak = abs(m) >= T and np.sign(h1) == np.sign(h2) == np.sign(m) and min(abs(h1), abs(h2)) >= T / 2
            rows.append({"요소": name, "col": col, "최근": round(m, 4), "반쪽": [round(h1, 4), round(h2, 4)], f"ρ({anchor})": round(rho, 2),
                         "약함 기준": bool(weak), "겹침": bool(abs(rho) >= RHO)})
        keep = []
        for r in sorted([r for r in rows if r["약함 기준"] and not r["겹침"]], key=lambda r: -abs(r["최근"])):
            if all(abs(float(rk[r["col"]].corr(rk[k["col"]]))) < RHO for k in keep):
                keep.append(r)
        res[reg] = {"기준 요소": anchor, "후보": rows, "남김": [{"요소": k["요소"], "방향": "높을수록 좋음" if k["최근"] > 0 else "높을수록 나쁨",
                                                          "최근": k["최근"], "반쪽": k["반쪽"]} for k in keep]}
    (results_dir() / "holding_weak_tier.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for reg, v in res.items():
        print("==", reg, "기준", v["기준 요소"])
        for r in v["후보"]:
            if r["약함 기준"]:
                print(f"   {r['요소']:14s} 최근 {r['최근']*100:+.2f} 반쪽 {r['반쪽'][0]*100:+.2f}/{r['반쪽'][1]*100:+.2f} ρ {list(r.values())[4]} {'겹침' if r['겹침'] else ''}")
        print("   남김:", [(k["요소"], k["방향"]) for k in v["남김"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
