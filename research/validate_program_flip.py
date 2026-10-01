"""복귀일 프로그램 매도 압력 반전 — 검증이력 9.51 사전 등록 기준 그대로.

  python research/validate_program_flip.py   → results/program_flip_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
HIST = "--hist" in _sys.argv         # 9.79: 2016~ 이력 확장 재판정(규칙·기준 그대로, 기간만) — 결과 파일에 _hist
SUF = "_hist" if HIST else ""
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
import validate_signal_combo as sc
import validate_stops as vs
from box_rules import BOX, FAIL_RECOVER, t2_flags
from collect_program import load_program
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

COST = vb.COST
SPLIT = pd.Timestamp("2021-06-01" if HIST else "2025-10-01")
LOOK = 10


def events(code: str, reg: pd.Series) -> list[dict]:
    bars, prog = load_bars(LONG_HISTORY, code), load_program(code)
    if bars is None or prog is None or len(bars) < 300 or prog.empty:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    t2 = t2_flags(bars)
    p = prog.reindex(bars.index)
    net, val = p["prog_net"].to_numpy(float), p["value"].to_numpy(float)
    rg = reg.reindex(bars.index).ffill().to_numpy()
    first = max(int(bars.index.searchsorted(prog.index.min())) + LOOK + 5, BOX + 5)
    out, busy = [], -1
    for b in range(first, len(c)):
        if b <= busy or np.isnan(L[b]) or not c[b] < L[b]:
            continue
        r = next((k for k in range(b + 1, min(b + 1 + FAIL_RECOVER, len(c))) if c[k] > L[b]), None)
        if r is None or r + 1 >= len(c):
            continue
        e, tgt, stop = r + 1, H[b], l[b:r + 1].min()
        if o[e] <= stop or o[e] >= tgt:
            continue
        sim = vs.simulate(o, h, l, c, e, stop, tgt, False)
        if sim is None:
            continue
        busy = e + sim[2] - 1
        w_net, w_val = net[r - LOOK:r], val[r - LOOK:r]
        if np.isnan(w_net).any() or np.isnan(net[r]) or np.nansum(w_val) <= 0:
            continue
        sells = w_net[w_net < 0]
        pressure = (w_net < 0).sum() >= 6 and w_net.sum() < 0
        flip = bool(pressure and (net[r] >= 0 or (len(sells) and abs(net[r]) < 0.1 * abs(sells.mean()))))
        sim3 = any(l[j] <= stop for j in range(e, min(e + 3, len(c))))
        out.append({"code": code, "entry_date": bars.index[e], "win": float(sim[0] - COST > 0),
                    "R": (sim[0] - COST) / max(1 - stop / o[e], 0.01), "whip3": float(sim3),
                    "t2": float(t2[b]), "reg": rg[b - 1], "reg_side": float(rg[b - 1] == "횡보·전환"), "reg_down": float(rg[b - 1] == "하락장"),
                    "depth": min(1 - stop / L[b], 0.12), "cpos": (c[r] - l[r]) / (h[r] - l[r]) if h[r] > l[r] else np.nan,
                    "flip": float(flip), "pressure": float(pressure), "cum": float(w_net.sum() / w_val.sum())})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    d = pd.DataFrame([ev for code in codes for ev in events(code, reg)]).dropna(subset=["cpos"])
    d["flip_nonbull"] = d["flip"] * (d["reg"] != "상승장")
    d.to_csv(results_dir() / f"program_flip_events{SUF}.csv", index=False)
    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3), "3일 안 손절": round(float(g.whip3.mean()), 3)}
    res = {"n": int(len(d)), "기간": [str(d.entry_date.min().date()), str(d.entry_date.max().date())],
           "서술": {"매도 압력 후 반전": s(d[d.flip == 1]), "매도 압력, 반전 없음": s(d[(d.pressure == 1) & (d.flip == 0)]),
                  "매도 압력 없음": s(d[d.pressure == 0])},
           "장세별 반전 vs 압력만": {r: [s(g[g.flip == 1]), s(g[(g.pressure == 1) & (g.flip == 0)])] for r, g in d.groupby("reg")}}
    base = ["t2", "reg_side", "reg_down", "depth", "cpos"]
    train, test = d[d.entry_date < SPLIT], d[d.entry_date >= SPLIT]
    res["n_train"], res["n_test"] = int(len(train)), int(len(test))
    for name, cols in (("반전", ["flip"]), ("반전 × 상승장 아님", ["flip", "flip_nonbull"]), ("누적 매도 강도", ["cum"])):
        r = sc.compare(train, test, base, base + cols, rng)
        r["verdict"] = "남김" if r["ci"][0] > 0 and r["logloss_new"] < r["logloss_base"] else "뺌"
        res[name] = r
    (results_dir() / f"program_flip_validation{SUF}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
