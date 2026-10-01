"""돌파 유지 확인 후 매수 — 검증이력 9.56 사전 등록 기준 그대로.

  python research/validate_breakout_hold.py   → results/breakout_hold_validation.json
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
HOLDS = (5, 10)


def events(code: str, reg: pd.Series, rng) -> tuple[list[dict], int]:
    bars = load_bars(LONG_HISTORY, code)
    if bars is None or len(bars) < 400:
        return [], 0
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    Hs = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    rg = reg.reindex(bars.index).ffill().to_numpy()
    first = max(int(bars.index.searchsorted(START)), 60)
    lo_c, hi_c = first, len(c) - vbt.HOLD
    out, n_first = [], 0
    for n in HOLDS:
        busy = -1
        for t1 in range(first, len(c) - n - 2):
            if np.isnan(Hs[t1]) or not (c[t1] > Hs[t1] and c[t1 - 1] <= Hs[t1 - 1]):
                continue
            if n == HOLDS[0]:
                n_first += 1
            H = Hs[t1]
            if (c[t1 + 1:t1 + n + 1] < H).any():
                continue
            tc = t1 + n
            if tc <= busy:
                continue
            e, stop = tc + 1, H - atr[tc]
            if not stop < o[e]:
                continue
            sim = vbt.simulate(o, h, l, c, ma20, e, stop)
            if sim is None:
                continue
            busy = e + sim[2] - 1
            sp = 1 - stop / o[e]
            ctrl = np.mean([vbt.simulate(o, h, l, c, ma20, int(d), o[d] * (1 - sp))[0] for d in rng.integers(lo_c, hi_c, 10)])
            out.append({"code": code, "hold": n, "entry_date": bars.index[e], "reg": rg[tc], "ret": sim[0], "stop_pct": sp,
                        "ctrl": float(ctrl), "win": float(sim[0] - COST > 0), "R": (sim[0] - COST) / max(sp, 0.01),
                        "run": c[tc] / H - 1})
    return out, n_first


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    rows, n_first = [], 0
    for code in codes:
        ev, nf = events(code, reg, rng)
        rows += ev
        n_first += nf
    hd = pd.DataFrame(rows)
    hd.to_csv(results_dir() / "breakout_hold_events.csv", index=False)
    fb = pd.read_csv(results_dir() / "breakout_edge_events.csv", parse_dates=["entry_date"])
    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "초과": round(float((g.ret - g.ctrl).mean()), 4)}
    res = {"첫 돌파 수": n_first}
    for n in HOLDS:
        h = hd[hd.hold == n]
        r = {"확인 통과 사건": int(len(h)), "장세별": {g: {"확인 후": s(h[h.reg == g]), "첫 돌파": s(fb[fb.reg == g])}
                                                 for g in ("상승장", "횡보·전환", "하락장")}}
        hu, fu = h[h.reg == "상승장"], fb[fb.reg == "상승장"]
        tr = vr.diff_ci(hu[hu.entry_date < SPLIT], fu[fu.entry_date < SPLIT], rng)
        te = vr.diff_ci(hu[hu.entry_date >= SPLIT], fu[fu.entry_date >= SPLIT], rng)
        r["상승장 확인 후 − 첫 돌파 R"] = {"맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te]}
        r["verdict"] = ("채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
                        "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함")
        res[f"{n}일 유지"] = r
    (results_dir() / "breakout_hold_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
