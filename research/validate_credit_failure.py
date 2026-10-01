"""신용잔고 급감 동반 가짜 이탈 — 검증이력 9.42 사전 등록 기준 그대로.

  python research/validate_credit_failure.py   → results/credit_failure_validation.json
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
import validate_stops as vs
from box_rules import BOX, FAIL_RECOVER, MAX_HOLD, t2_flags
from collect_credit import load_credit
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START = pd.Timestamp("2016-01-11" if HIST else "2024-10-15")
COST = vb.COST


def events(code: str, rng) -> list[dict]:
    bars = load_bars(LONG_HISTORY, code)
    cr = load_credit(code)
    if bars is None or cr is None or len(bars) < 300 or cr.empty:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    t2 = t2_flags(bars)
    loan = cr["loan_shares"].reindex(bars.index).to_numpy(float)
    chg5 = pd.Series(loan).pct_change(5).to_numpy()
    first = max(int(bars.index.searchsorted(START)), BOX + 70)
    lo_c, hi_c = int(bars.index.searchsorted(START)), len(c) - MAX_HOLD
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
        base = np.nanmean(loan[b - 5:b])
        if not (base > 0) or np.isnan(loan[b]):
            continue
        hist = chg5[b - 65:b - 5]
        hist = hist[~np.isnan(hist)]
        d_b = loan[b] / base - 1
        z = (d_b - hist.mean()) / hist.std() if len(hist) > 20 and hist.std() > 0 else np.nan
        d_r = loan[r] / base - 1 if not np.isnan(loan[r]) else np.nan
        sp, tp = 1 - stop / o[e], tgt / o[e] - 1
        ctrl = np.mean([vs.simulate(o, h, l, c, int(d), o[d] * (1 - sp), o[d] * (1 + tp), False)[0]
                        for d in rng.integers(lo_c, hi_c, 10)])
        out.append({"code": code, "entry_date": bars.index[e], "t2": bool(t2[b]), "d_b": d_b, "z": z, "d_r": d_r,
                    "ret": sim[0], "stop_pct": sp, "ctrl": float(ctrl)})
    return out


def stats(g: pd.DataFrame) -> dict:
    net = g["ret"] - COST
    return {"n": int(len(g)), "win": round(float((net > 0).mean()), 3), "R": round(float((net / g.stop_pct.clip(lower=0.01)).mean()), 3),
            "excess": round(float((g["ret"] - g["ctrl"]).mean()), 4)}


def diff_ci(a: pd.DataFrame, b: pd.DataFrame, rng) -> tuple[float, float, float]:
    """(a − b) 평균 R 차이, 20일 블록 부트스트랩."""
    x = pd.concat([a.assign(g=1), b.assign(g=0)])
    x["R"] = (x["ret"] - COST) / x["stop_pct"].clip(lower=0.01)
    x["block"] = x["entry_date"].rank(method="dense").astype(int) // vb.BLOCK
    agg = np.array([(y.loc[y.g == 1, "R"].sum(), (y.g == 1).sum(), y.loc[y.g == 0, "R"].sum(), (y.g == 0).sum())
                    for _, y in x.groupby("block")], float)
    idx = rng.integers(0, len(agg), (vb.N_BOOT, len(agg)))
    s = agg[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    point = x.loc[x.g == 1, "R"].mean() - x.loc[x.g == 0, "R"].mean()
    return float(point), float(lo), float(hi)


def compare(d: pd.DataFrame, col: str, rng) -> dict:
    d = d.dropna(subset=[col])
    cut = d[col].quantile(0.3)
    down, up = d[d[col] <= cut], d[d[col] >= 0]
    p, lo, hi = diff_ci(down, up, rng) if len(down) > 5 and len(up) > 5 else (np.nan, np.nan, np.nan)
    return {"cut": round(float(cut), 4), "급감": stats(down), "증가·변동없음": stats(up), "R차이": round(p, 3), "ci": [round(lo, 3), round(hi, 3)]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    d = pd.DataFrame([ev for code in codes for ev in events(code, rng)])
    d.to_csv(results_dir() / f"credit_failure_events{SUF}.csv", index=False)
    res = {"events": int(len(d)), "t2_events": int(d["t2"].sum())}
    for scope, x in (("T2", d[d.t2]), ("T2 없음(추가 확인)", d[~d.t2]), ("전체", d)):
        res[scope] = {"주(이탈일까지)": compare(x, "d_b", rng), "참고(복귀일까지)": compare(x, "d_r", rng)}
        z = x.dropna(subset=["z"])
        if len(z):
            zd, zr = z[z.z < -1.5], z[z.z >= -1.5]
            p, lo, hi = diff_ci(zd, zr, rng)
            res[scope]["z<-1.5"] = {"급감": stats(zd), "나머지": stats(zr), "R차이": round(p, 3), "ci": [round(lo, 3), round(hi, 3)]}
    t = res["T2"]["주(이탈일까지)"]
    small = min(t["급감"]["n"], t["증가·변동없음"]["n"]) < 50
    res["verdict"] = ("판단 보류(T² 표본 부족)" if small else
                      "채택" if t["R차이"] >= 0.2 and t["ci"][0] > 0 else "방향만 일치" if t["R차이"] >= 0.2 else "채택 안 함")
    (results_dir() / f"credit_failure_validation{SUF}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
