"""돌파 승률 모형 — 엣지 찾기. 검증이력 9.52 사전 등록 기준 그대로.

  python research/validate_breakout_edge.py   → results/breakout_edge_validation.json, results/breakout_edge_events.csv
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
import validate_signal_combo as sc
import validate_stops as vs
from box_rules import BOX, t2_flags
from v2_compression import compression_frame
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST = vb.COST
CANDS = {"돌파 강도": "strength", "돌파 캔들 종가 위치": "bcpos", "돌파일 T²": "t2", "직전 응축": "comp",
         "종목 변동성(250일)": "vol250", "박스 폭": "width"}


def events(code: str, reg: pd.Series, rng) -> list[dict]:
    bars = load_bars(LONG_HISTORY, code)
    if bars is None or len(bars) < 400:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    Lb = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    vol250 = pd.Series(np.log(c)).diff().rolling(250).std().to_numpy()
    t2 = t2_flags(bars)
    comp = compression_frame(bars)["compressed"].fillna(False).to_numpy()
    rg = reg.reindex(bars.index).ffill().to_numpy()
    first = max(int(bars.index.searchsorted(START)), 260)
    lo_c, hi_c = first, len(c) - vbt.HOLD
    out, busy = [], -1
    for t in range(first, len(c) - 1):
        if t <= busy or np.isnan(H[t]) or not (c[t] > H[t] and c[t - 1] <= H[t - 1]):
            continue
        e, stop = t + 1, H[t] - atr[t]
        if not stop < o[e]:
            continue
        sim = vbt.simulate(o, h, l, c, ma20, e, stop)
        if sim is None:
            continue
        busy = e + sim[2] - 1
        sp = 1 - stop / o[e]
        ctrl = np.mean([vbt.simulate(o, h, l, c, ma20, int(d), o[d] * (1 - sp))[0] for d in rng.integers(lo_c, hi_c, 10)])
        out.append({"code": code, "entry_date": bars.index[e], "reg": rg[t], "ret": sim[0], "stop_pct": sp, "ctrl": float(ctrl),
                    "win": float(sim[0] - COST > 0), "R": (sim[0] - COST) / max(sp, 0.01),
                    "strength": c[t] / H[t] - 1, "bcpos": (c[t] - l[t]) / (h[t] - l[t]) if h[t] > l[t] else np.nan,
                    "t2": float(t2[t]), "comp": float(comp[max(0, t - 10):t].any()), "vol250": vol250[t], "width": H[t] / Lb[t] - 1})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    d = pd.DataFrame([ev for code in codes for ev in events(code, reg, rng)])
    d.to_csv(results_dir() / "breakout_edge_events.csv", index=False)
    d = d.dropna(subset=list(CANDS.values()) + ["reg"])
    d["reg_side"] = (d["reg"] == "횡보·전환").astype(float)
    d["reg_down"] = (d["reg"] == "하락장").astype(float)
    base = ["reg_side", "reg_down"]
    res = {"n": int(len(d)), "장세별": {}}
    for r, g in d.groupby("reg"):
        x = {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
             "초과": round(float((g.ret - g.ctrl).mean()), 4)}
        for name, col in CANDS.items():
            if col in ("t2", "comp"):
                parts = [("예", g[g[col] == 1]), ("아니오", g[g[col] == 0])]
            else:
                q1, q2 = g[col].quantile([1 / 3, 2 / 3])
                parts = [("낮음", g[g[col] <= q1]), ("중간", g[(g[col] > q1) & (g[col] <= q2)]), ("높음", g[g[col] > q2])]
            x[name] = {k: f"n={len(p)} 승률 {p.win.mean():.0%} {p.R.mean():+.2f}R 초과 {(p.ret - p.ctrl).mean():+.2%}" for k, p in parts}
        res["장세별"][r] = x
    train, test = d[d.entry_date < SPLIT], d[d.entry_date >= SPLIT]
    res["n_train"], res["n_test"] = int(len(train)), int(len(test))
    keep = []
    for name, col in CANDS.items():
        r = sc.compare(train, test, base, base + [col], rng)
        r["verdict"] = "남김" if r["ci"][0] > 0 and r["logloss_new"] < r["logloss_base"] else "뺌"
        res[name] = r
        if r["verdict"] == "남김":
            keep.append(col)
    if keep:
        res["남긴 변수 전부"] = sc.compare(train, test, base, base + keep, rng) | {"vars": keep}
    (results_dir() / "breakout_edge_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
