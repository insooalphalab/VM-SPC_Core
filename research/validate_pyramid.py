"""나눠 사기(피라미딩) 30/30/40 — 검증이력 9.74 사전 등록 그대로.

  python research/validate_pyramid.py   → results/pyramid_validation.json
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
import validate_rebreakout as vr
from box_rules import BOX
from validate_breakout_exits import EXITS, simulate
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
COST, E2 = vb.COST, EXITS["E2 50일선"]
W = (0.3, 0.3, 0.4)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "breakout_exits_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    tranches, events = [], []
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        ma50 = pd.Series(c).rolling(50).mean().to_numpy()
        for _, x in g.iterrows():
            e = int(b.index.searchsorted(x.entry_date))
            t = e - 1
            if e + 2 >= len(c):
                continue
            H = h[t - BOX:t].max()
            stop0 = c[t] - atr[t]
            ret, hold = simulate(o, h, l, c, ma50, atr, e, stop0, E2)
            px = o[e] * (1 + ret)                                          # 모든 몫이 같은 날 같은 값에 나간다
            buys = [(1, o[e])]
            z = (c[e] - c[t]) / max(c[t] - H, 0.1 * atr[t])
            if hold >= 2 and z > 1 and o[e + 1] > stop0:
                buys.append((2, o[e + 1]))
                if hold >= 3 and c[e + 1] > c[e] and o[e + 2] > stop0:
                    buys.append((3, o[e + 2]))
            pnl = risk = 0.0
            for k, p in buys:
                r_k = (px / p - 1 - COST) / max(1 - stop0 / p, 0.01)
                tranches.append({"code": code, "entry_date": x.entry_date, "단계": k, "R": r_k})
                pnl += W[k - 1] * (px / p - 1 - COST)
                risk += W[k - 1] * max(1 - stop0 / p, 0.01)
            p0_ret = px / o[e] - 1 - COST
            events.append({"entry_date": x.entry_date, "stages": len(buys), "pyr_pnl": pnl, "pyr_risk": risk,
                           "p0_pnl": p0_ret, "p0_risk": max(1 - stop0 / o[e], 0.01)})
    T = pd.DataFrame(tranches)
    E = pd.DataFrame(events)
    T.to_csv(results_dir() / "pyramid_tranches.csv", index=False)
    fit_e, test_e = E.entry_date < SPLIT, E.entry_date >= SPLIT
    p0 = T[T["단계"] == 1]

    def ratio(x, a, b):
        return round(float(x[a].sum() / x[b].sum()), 3)

    res = {"n": int(len(E)), "단계 발동 비율": {k: round(float((E.stages >= k).mean()), 3) for k in (1, 2, 3)},
           "몫별 R": {f"{k}차": {"n": int((T["단계"] == k).sum()), "R": round(float(T.R[T["단계"] == k].mean()), 3),
                                "승률": round(float((T.R[T["단계"] == k] > 0).mean()), 3)} for k in (1, 2, 3)},
           "위험당 R (손익 합 ÷ 위험 합)": {"P0 한 번에": [ratio(E, "p0_pnl", "p0_risk"), ratio(E[fit_e], "p0_pnl", "p0_risk"), ratio(E[test_e], "p0_pnl", "p0_risk")],
                                       "30/30/40": [ratio(E, "pyr_pnl", "pyr_risk"), ratio(E[fit_e], "pyr_pnl", "pyr_risk"), ratio(E[test_e], "pyr_pnl", "pyr_risk")],
                                       "_순서": "전체 / 맞춤 / 예측"},
           "같은 자본 거래당 평균 손익": {"P0": round(float(E.p0_pnl.mean()), 4), "30/30/40": round(float(E.pyr_pnl.mean()), 4)},
           "거래당 평균 위험(자본 대비)": {"P0": round(float(E.p0_risk.mean()), 4), "30/30/40": round(float(E.pyr_risk.mean()), 4)},
           "최악 거래 손익": {"P0": round(float(E.p0_pnl.min()), 4), "30/30/40": round(float(E.pyr_pnl.min()), 4)}}
    for k in (2, 3):
        a = T[T["단계"] == k]
        tr = vr.diff_ci(a[a.entry_date < SPLIT], p0[p0.entry_date < SPLIT], rng)
        te = vr.diff_ci(a[a.entry_date >= SPLIT], p0[p0.entry_date >= SPLIT], rng)
        res[f"{k}차 몫 R − 한 번에 R"] = {"맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                       "verdict": "채택" if te[0] >= 0.10 and te[1] > 0 and tr[0] > 0 else
                                                  "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "pyramid_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
