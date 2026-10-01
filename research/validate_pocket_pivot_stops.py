"""포켓 피봇 손절 변형 — 검증이력 9.59 추가 사전 등록 그대로.

  python research/validate_pocket_pivot_stops.py   → results/pocket_pivot_stops_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

import validate_box as vb
import validate_breakout_trend as vbt
import validate_rebreakout as vr
import validate_stops as vs
from box_rules import BOX
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST = vb.COST
MODES = ("신호일 저가", "50일선", "진입가 − 1ATR")


def events(code: str, reg: pd.Series, rng) -> list[dict]:
    bars = load_bars(LONG_HISTORY, code)
    if bars is None or len(bars) < 400:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    ma50 = pd.Series(c).rolling(50).mean().to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    Hs = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    rg = reg.reindex(bars.index).ffill().to_numpy()
    first = max(int(bars.index.searchsorted(START)), 60)
    lo_c, hi_c = first, len(c) - vbt.HOLD
    out, busy = [], -1
    for t in range(first, len(c) - 1):
        if t <= busy or np.isnan(ma50[t]) or np.isnan(Hs[t]):
            continue
        if not (c[t] > ma50[t] and c[t] <= Hs[t] and c[t] > c[t - 1] and c[t] > o[t]):
            continue
        rng_t = h[t] - l[t]
        if rng_t <= 0 or (h[t] - c[t]) / rng_t >= 0.3:
            continue
        if not v[t - 10:t].mean() < v[t - 50:t - 10].mean():
            continue
        down = [v[k] for k in range(t - 10, t) if c[k] < c[k - 1]]
        if not down or not v[t] > max(down):
            continue
        e = t + 1
        stops = {"신호일 저가": l[t], "50일선": ma50[t], "진입가 − 1ATR": o[e] - atr[t]}
        row = {"code": code, "entry_date": bars.index[e], "reg": rg[t]}
        ok, hold = True, None
        days = rng.integers(lo_c, hi_c, 10)
        for m in MODES:
            s = stops[m]
            if not s < o[e]:
                ok = False
                break
            r = vbt.simulate(o, h, l, c, ma20, e, s)
            if r is None:
                ok = False
                break
            hold = hold or r[2]
            sp = 1 - s / o[e]
            ctrl = np.mean([vbt.simulate(o, h, l, c, ma20, int(d), o[d] * (1 - sp))[0] for d in days])
            row[f"{m}|R"] = (r[0] - COST) / max(sp, 0.005)
            row[f"{m}|win"] = float(r[0] - COST > 0)
            row[f"{m}|ex"] = r[0] - ctrl
            row[f"{m}|sp"] = sp
            R_ = row[f"{m}|R"]
        if not ok:
            continue
        busy = e + hold - 1
        out.append(row)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    pp = pd.DataFrame([ev for code in codes for ev in events(code, reg, rng)])
    pp.to_csv(results_dir() / "pocket_pivot_stops_events.csv", index=False)
    fb = pd.read_csv(results_dir() / "breakout_edge_events.csv", parse_dates=["entry_date"])
    fu = fb[fb.reg == "상승장"]
    res = {}
    for r in ("상승장", "횡보·전환", "하락장"):
        g = pp[pp.reg == r]
        res[r] = {m: {"n": int(len(g)), "승률": round(float(g[f"{m}|win"].mean()), 3), "R": round(float(g[f"{m}|R"].mean()), 3),
                      "초과": round(float(g[f"{m}|ex"].mean()), 4), "손절 거리": round(float(g[f"{m}|sp"].median()), 4),
                      "진 거래 평균 R": round(float(g.loc[g[f"{m}|R"] < 0, f"{m}|R"].mean()), 2)} for m in MODES}
    pu = pp[pp.reg == "상승장"]
    for m in MODES:
        a = pu.assign(R=pu[f"{m}|R"])
        tr = vr.diff_ci(a[a.entry_date < SPLIT], fu[fu.entry_date < SPLIT], rng)
        te = vr.diff_ci(a[a.entry_date >= SPLIT], fu[fu.entry_date >= SPLIT], rng)
        res[f"상승장 {m} − 첫 돌파 R"] = {"맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te],
                                       "verdict": "채택 후보(이력 확장 뒤 재확인)" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
                                                  "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "pocket_pivot_stops_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
