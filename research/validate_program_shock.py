"""프로그램매매 비정상 충격(거래대금 대비 순매수 z) — 검증이력 9.47 사전 등록 기준 그대로.

  python research/validate_program_shock.py   → results/program_shock_validation.json
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
import validate_breakout_trend as vbt
import validate_stops as vs
from box_rules import BOX, FAIL_RECOVER, MAX_HOLD, t2_flags
from collect_program import load_program
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

COST = vb.COST
Z_WIN, Z_CUT = 60, 2.0


def share_z(bars: pd.DataFrame, prog: pd.DataFrame) -> np.ndarray:
    p = prog.reindex(bars.index)
    share = (p["prog_net"] / p["value"]).to_numpy(float)
    s = pd.Series(share)
    mu, sd = s.rolling(Z_WIN, min_periods=40).mean().shift(1), s.rolling(Z_WIN, min_periods=40).std().shift(1)
    return ((s - mu) / sd).to_numpy()


def events(code: str, market: str, rng) -> tuple[list[dict], list[dict]]:
    bars, prog = load_bars(LONG_HISTORY, code), load_program(code)
    if bars is None or prog is None or len(bars) < 300 or prog.empty:
        return [], []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    z = share_z(bars, prog)
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    t2 = t2_flags(bars)
    first = max(int(bars.index.searchsorted(prog.index.min())) + Z_WIN, BOX + 15)
    lo_c, hi_c = int(bars.index.searchsorted(prog.index.min())), len(c) - max(MAX_HOLD, vbt.HOLD)
    fail, busy = [], -1
    for b in range(first, len(c)):                         # H1 가짜 이탈(9.18 규칙)
        if b <= busy or np.isnan(L[b]) or not c[b] < L[b] or np.isnan(z[b]):
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
        sp, tp = 1 - stop / o[e], tgt / o[e] - 1
        ctrl = np.mean([vs.simulate(o, h, l, c, int(d), o[d] * (1 - sp), o[d] * (1 + tp), False)[0] for d in rng.integers(lo_c, hi_c, 10)])
        fail.append({"code": code, "market": market, "entry_date": bars.index[e], "t2": bool(t2[b]), "z": z[b],
                     "ret": sim[0], "stop_pct": sp, "ctrl": float(ctrl)})
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    brk, busy = [], -1
    for t in range(first, len(c) - 1):                     # H2 추세 추종 돌파(9.40 규칙)
        if t <= busy or np.isnan(z[t]) or not (c[t] > H[t] and c[t - 1] <= H[t - 1]):
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
        brk.append({"code": code, "market": market, "entry_date": bars.index[e], "z": z[t], "ret": sim[0], "stop_pct": sp, "ctrl": float(ctrl)})
    return fail, brk


def stats(g: pd.DataFrame) -> dict:
    net = g["ret"] - COST
    return {"n": int(len(g)), "win": round(float((net > 0).mean()), 3), "R": round(float((net / g.stop_pct.clip(lower=0.01)).mean()), 3),
            "excess": round(float((g["ret"] - g["ctrl"]).mean()), 4)}


def diff_ci(d: pd.DataFrame, flag: pd.Series, col: str, rng) -> tuple[float, float, float]:
    x = d.assign(g=flag.astype(int), v=col(d) if callable(col) else d[col])
    x["block"] = x["entry_date"].rank(method="dense").astype(int) // vb.BLOCK
    agg = np.array([(y.loc[y.g == 1, "v"].sum(), (y.g == 1).sum(), y.loc[y.g == 0, "v"].sum(), (y.g == 0).sum())
                    for _, y in x.groupby("block")], float)
    idx = rng.integers(0, len(agg), (vb.N_BOOT, len(agg)))
    s = agg[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(x.loc[x.g == 1, "v"].mean() - x.loc[x.g == 0, "v"].mean()), float(lo), float(hi)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    kospi = set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
    kosdaq = set(vs._codes("oos_kosdaq_codes.json"))
    F, B = [], []
    for code in sorted(kospi | kosdaq):
        f, b = events(code, "코스닥" if code in kosdaq else "코스피·센서", rng)
        F += f
        B += b
    fd, bd = pd.DataFrame(F), pd.DataFrame(B)
    fd.to_csv(results_dir() / f"program_shock_failure{SUF}.csv", index=False)
    bd.to_csv(results_dir() / f"program_shock_breakout{SUF}.csv", index=False)
    R = lambda d: (d["ret"] - COST) / d["stop_pct"].clip(lower=0.01)
    res = {}
    for scope, d in (("가짜 이탈 T²", fd[fd.t2]), ("가짜 이탈 전체", fd)):
        shock = d["z"] < -Z_CUT
        p, lo, hi = diff_ci(d, shock, R, rng)
        res[scope] = {"충격(z<-2)": stats(d[shock]), "나머지": stats(d[~shock]), "R차이": round(p, 3), "ci": [round(lo, 3), round(hi, 3)],
                      "시장별 R차이": {m: round(float(R(g[g.z < -Z_CUT]).mean() - R(g[g.z >= -Z_CUT]).mean()), 3)
                                    for m, g in d.groupby("market")},
                      "verdict": "채택" if p >= 0.2 and lo > 0 else "방향만 일치" if p >= 0.2 else "채택 안 함"}
    shock = bd["z"] > Z_CUT
    ex = lambda d: d["ret"] - d["ctrl"]
    p, lo, hi = diff_ci(bd, shock, ex, rng)
    res["돌파"] = {"충격(z>+2)": stats(bd[shock]), "나머지": stats(bd[~shock]), "초과차이": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                 "시장별 초과차이": {m: round(float(ex(g[g.z > Z_CUT]).mean() - ex(g[g.z <= Z_CUT]).mean()), 4) for m, g in bd.groupby("market")},
                 "verdict": "채택" if p >= 0.01 and lo > 0 else "방향만 일치" if p >= 0.01 else "채택 안 함"}
    (results_dir() / f"program_shock_validation{SUF}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
