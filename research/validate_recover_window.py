"""복귀 기한(3일) 민감도 — 검증이력 9.49 사전 등록 기준 그대로.

  python research/validate_recover_window.py   → results/recover_window_validation.json
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
import validate_stops as vs
from box_rules import BOX, MAX_HOLD, t2_flags
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START = pd.Timestamp("2016-01-01")
WINDOWS = (1, 2, 3, 5, 7, 10)
COST = vb.COST


def events(code: str, rng) -> list[dict]:
    bars = load_bars(LONG_HISTORY, code)
    if bars is None or len(bars) < 300:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    t2 = t2_flags(bars)
    lo_c, hi_c = int(bars.index.searchsorted(START)), len(c) - MAX_HOLD
    out = []
    for w in WINDOWS:
        busy = -1
        for b in range(max(int(bars.index.searchsorted(START)), BOX + 5), len(c)):
            if b <= busy or np.isnan(L[b]) or not c[b] < L[b]:
                continue
            r = next((k for k in range(b + 1, min(b + 1 + w, len(c))) if c[k] > L[b]), None)
            if r is None or r + 1 >= len(c):
                continue
            e, tgt, stop = r + 1, H[b], l[b:r + 1].min()
            if o[e] <= stop or o[e] >= tgt:
                continue
            sim = vs.simulate(o, h, l, c, e, stop, tgt, False)
            if sim is None:
                continue
            busy = e + sim[2] - 1
            if not t2[b]:
                continue
            sp, tp = 1 - stop / o[e], tgt / o[e] - 1
            ctrl = np.mean([vs.simulate(o, h, l, c, int(d), o[d] * (1 - sp), o[d] * (1 + tp), False)[0] for d in rng.integers(lo_c, hi_c, 10)])
            out.append({"code": code, "w": w, "rec_days": r - b, "entry_date": bars.index[e], "ret": sim[0], "stop_pct": sp, "ctrl": float(ctrl)})
    return out


def stats(g: pd.DataFrame) -> dict:
    net = g["ret"] - COST
    return {"n": int(len(g)), "win": round(float((net > 0).mean()), 3), "R": round(float((net / g.stop_pct.clip(lower=0.01)).mean()), 3),
            "excess": round(float((g["ret"] - g["ctrl"]).mean()), 4)}


def boot_ci(g: pd.DataFrame, rng) -> list[float]:
    ex = (g["ret"] - g["ctrl"]).to_numpy()
    blk = (g["entry_date"].rank(method="dense").astype(int) // vb.BLOCK).to_numpy()
    lo, hi = vb.boot_mean(ex, blk, rng)
    return [round(lo, 4), round(hi, 4)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    sets = {"센서(참고)": sorted(sensor_universe()), "코스피 1~200(참고)": vs._codes("oos_codes.json"),
            "코스피 다음 200": vs._codes("oos_kospi2_codes.json"), "코스닥 1~200": vs._codes("oos_kosdaq_codes.json")}
    res, frames = {}, {}
    for u, codes in sets.items():
        d = pd.DataFrame([ev for code in codes for ev in events(code, rng)])
        frames[u] = d
        r = {f"{w}일": stats(d[d.w == w]) for w in WINDOWS}
        w10 = d[d.w == 10]
        for lab, lo_, hi_ in (("늘어난 구간 4~5일", 4, 5), ("6~7일", 6, 7), ("8~10일", 8, 10)):
            r[lab] = stats(w10[w10.rec_days.between(lo_, hi_)])
        res[u] = r
        print(f"\n== {u}")
        for k, s in r.items():
            print(f"  {k:12s} n={s['n']:5d} 승률 {s['win']:.0%} {s['R']:+.2f}R 무작위 대비 {s['excess']:+.2%}")
    judge = ("코스피 다음 200", "코스닥 1~200")
    verdict = "3일 유지"
    for w in WINDOWS:
        if w == 3:
            continue
        better = all(res[u][f"{w}일"]["excess"] - res[u]["3일"]["excess"] >= 0.003 for u in judge)
        if not better:
            continue
        if w > 3:
            pool = pd.concat([frames[u][(frames[u].w == w) & (frames[u].rec_days > 3)] for u in judge])
        else:
            pool = pd.concat([frames[u][(frames[u].w == w)] for u in judge])
        ci = boot_ci(pool, rng)
        res[f"후보 {w}일"] = {"ci_늘어난구간": ci}
        if ci[0] > 0:
            verdict = f"{w}일로 변경 후보"
    res["verdict"] = verdict
    (results_dir() / "recover_window_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n판정:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
