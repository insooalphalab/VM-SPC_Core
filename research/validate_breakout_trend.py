"""추세 추종형 돌파(박스 상단 돌파 → 짧은 손절 + 20일선 청산) — 검증이력 9.40 사전 등록 기준 그대로.

  python research/validate_breakout_trend.py   → results/breakout_trend_validation.json
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
from box_rules import BOX
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START = pd.Timestamp("2016-01-01")
HOLD = 60
COST = vb.COST
JUDGE = ("코스피 1~200", "코스피 다음 200")


def simulate(o, h, l, c, ma20, e, stop):
    """e일 시가 진입 → (수익, 결과, 보유일). 손절(갭이면 시가) · 종가 < 20일선 · HOLD일."""
    if e + HOLD - 1 >= len(c):
        return None
    entry = o[e]
    for j in range(e, e + HOLD):
        if j > e and o[j] <= stop:
            return o[j] / entry - 1, "stop", j - e + 1
        if l[j] <= stop:
            return stop / entry - 1, "stop", j - e + 1
        if c[j] < ma20[j]:
            return c[j] / entry - 1, "ma20", j - e + 1
    return c[e + HOLD - 1] / entry - 1, "timeout", HOLD


def run(codes: list[str], reg: pd.Series, rng) -> pd.DataFrame:
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
        ma20 = pd.Series(c).rolling(20).mean().to_numpy()
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        first, lo_c, hi_c = int(bars.index.searchsorted(START)), int(bars.index.searchsorted(START)), len(c) - HOLD
        busy = -1
        for t in range(max(first, BOX + 15), len(c) - 1):
            if t <= busy or not (c[t] > H[t] and c[t - 1] <= H[t - 1]):
                continue
            e, stop = t + 1, H[t] - atr[t]
            if not stop < o[e]:
                continue
            sim = simulate(o, h, l, c, ma20, e, stop)
            if sim is None:
                continue
            busy = e + sim[2] - 1
            sp = 1 - stop / o[e]
            ctrl = [simulate(o, h, l, c, ma20, int(d), o[d] * (1 - sp))[0] for d in rng.integers(lo_c, hi_c, vb.N_CTRL)]
            rows.append({"code": code, "entry_date": bars.index[e], "ret": sim[0], "how": sim[1], "hold": sim[2],
                         "stop_pct": sp, "ctrl_ret": float(np.mean(ctrl))})
    d = pd.DataFrame(rows)
    d["reg"] = reg.iloc[reg.index.searchsorted(d["entry_date"]) - 1].to_numpy()
    d["block"] = reg.index.searchsorted(d["entry_date"]) // vb.BLOCK
    d["net"] = d["ret"] - COST
    d["ex"] = d["ret"] - d["ctrl_ret"]
    return d


def stats(g: pd.DataFrame, rng) -> dict:
    lo, hi = vb.boot_mean(g["ex"].to_numpy(), g["block"].to_numpy(), rng)
    w, ls = g.net[g.net > 0], g.net[g.net <= 0]
    top = g.net.sort_values(ascending=False)
    k = max(1, len(g) // 10)
    return {"n": int(len(g)), "win": round(float((g.net > 0).mean()), 3), "avg_win": round(float(w.mean()), 4),
            "avg_loss": round(float(ls.mean()), 4), "payoff": round(float(w.mean() / -ls.mean()), 2),
            "net": round(float(g.net.mean()), 4), "R": round(float((g.net / g.stop_pct.clip(lower=0.01)).mean()), 3),
            "top10_share": round(float(top.iloc[:k].sum() / g.net.sum()), 2) if g.net.sum() > 0 else None,
            "hold": round(float(g.hold.mean()), 1), "stop_pct": round(float(g.stop_pct.median()), 4),
            "ex": round(float(g.ex.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
            "exit": g["how"].value_counts(normalize=True).round(2).to_dict()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    sets = {"코스피 1~200": vs._codes("oos_codes.json"), "코스피 다음 200": vs._codes("oos_kospi2_codes.json"),
            "센서(참고)": sorted(sensor_universe()), "코스닥 밖(참고)": vs._codes("oos_kosdaq_codes.json")}
    res, verdicts = {}, {}
    for u, codes in sets.items():
        d = run(codes, reg, rng)
        r = {"전체": stats(d, rng)}
        for rg in ("상승장", "횡보·전환", "하락장"):
            r[rg] = stats(d[d.reg == rg], rng)
        other = stats(d[d.reg != "상승장"], rng)
        r["상승장 아님"] = other
        up = r["상승장"]
        gap = up["ex"] - other["ex"]
        r["verdict"] = ("재현" if up["ci"][0] > 0 and gap >= 0.005 else "방향만 일치" if up["ex"] > 0 and gap >= 0.005 else "채택 안 함")
        r["gap"] = round(gap, 4)
        res[u] = r
        print(f"\n== {u}  (상승장 − 그 외 {gap:+.2%} → {r['verdict']})")
        for k in ("전체", "상승장", "횡보·전환", "하락장"):
            s = r[k]
            print(f"  {k:5s} n={s['n']:5d} 승률 {s['win']:.0%} 이긴 {s['avg_win']:+.1%} 진 {s['avg_loss']:+.1%} 손익비 {s['payoff']:.1f} "
                  f"거래당 {s['net']:+.2%} {s['R']:+.2f}R 상위10% 몫 {s['top10_share']} 보유 {s['hold']:.0f}일 손절 {s['stop_pct']:.1%} | "
                  f"무작위 대비 {s['ex']:+.2%} [{s['ci'][0]:+.2%}, {s['ci'][1]:+.2%}]")
    v = [res[u]["verdict"] for u in JUDGE]
    res["최종"] = "유망" if all(x == "재현" for x in v) else "약함" if all(x in ("재현", "방향만 일치") for x in v) else "채택 안 함"
    print("\n최종:", res["최종"])
    (results_dir() / "breakout_trend_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
