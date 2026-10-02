"""거래량 가뭄 양봉(매물 잠김) vs 거래량 폭발 양봉 — 검증이력 9.101 사전 등록 그대로.

  python research/validate_volume_drought.py   → results/volume_drought.json
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
from universe_all import validation_codes
from validate_breakout_exits import EXITS, simulate
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

START, SPLIT, HALF, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
GAP = 5


def group(vx: float) -> str | None:
    return "A 가뭄" if vx <= 0.5 else "B 보통" if 0.8 <= vx <= 1.5 else "C 폭발" if vx >= 2.0 else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    idx = load_bars(LONG_HISTORY, "069500")
    reg = regime()
    rows = []
    for code in validation_codes():
        b = load_bars(LONG_HISTORY, code)
        if b is None or len(b) < 300:
            continue
        b = b[(b[["open", "high", "low", "close", "volume"]] > 0).all(1)]
        o, h, l, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        n = len(c)
        io = idx["open"].reindex(b.index).to_numpy(float)
        ic = idx["close"].reindex(b.index).to_numpy(float)
        va = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        H20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        ma = pd.Series(c).rolling(50).mean().to_numpy()
        first = int(b.index.searchsorted(START))
        last = -GAP
        for t in range(max(first, 60), n - 21):
            if t - last < GAP:
                continue
            up3 = c[t] / c[t - 1] - 1 >= 0.03 and c[t] > o[t]
            brk = c[t] > H20[t]
            if not (up3 or brk) or np.isnan(va[t]) or va[t] <= 0:
                continue
            g = group(v[t] / va[t])
            if g is None:
                continue
            last = t
            e = t + 1
            if not np.isfinite(io[e]) or not np.isfinite(ic[t + 20]):
                continue
            row = {"code": code, "entry_date": b.index[e], "reg": reg.get(b.index[t], np.nan), "g": g, "brk": bool(brk), "up3": bool(up3)}
            for k in (3, 5, 20):
                row[f"x{k}"] = (c[t + k] / o[e] - 1) - (ic[t + k] / io[e] - 1)
            if brk and not np.isnan(atr[t]) and not np.isnan(ma[t]) and o[e] > c[t] - atr[t]:
                stop0 = c[t] - atr[t]
                ret, _ = simulate(o, h, l, c, ma, atr, e, stop0, EXITS["E2 50일선"])
                row["Rb"] = (ret - vb.COST) / max(1 - stop0 / o[e], 0.01)
            rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(results_dir() / "volume_drought_events.csv", index=False)

    def cmp(x: pd.DataFrame, ga: str, gb: str, col: str, scale: float) -> dict:
        a, b = x[x.g == ga].assign(R=lambda z: z[col] * scale), x[x.g == gb].assign(R=lambda z: z[col] * scale)
        a, b = a.dropna(subset=["R"]), b.dropna(subset=["R"])
        per = lambda z, lo, hi: z[(z.entry_date >= lo) & (z.entry_date < hi)]
        out = {"n": [len(a), len(b)], "평균": [round(float(a.R.mean()), 2), round(float(b.R.mean()), 2)]}
        for pn, lo, hi in (("앞", START, SPLIT), ("최근", SPLIT, END)):
            aa, bb = per(a, lo, hi), per(b, lo, hi)
            out[pn] = [round(q, 2) for q in vr.diff_ci(aa, bb, rng)] if len(aa) >= 30 and len(bb) >= 30 else None
        out["반쪽"] = [round(float(per(a, SPLIT, HALF).R.mean() - per(b, SPLIT, HALF).R.mean()), 2),
                      round(float(per(a, HALF, END).R.mean() - per(b, HALF, END).R.mean()), 2)]
        return out

    def verdict(c: dict, T: float) -> str:
        o, r, h = c["앞"], c["최근"], c["반쪽"]
        if not o or not r:
            return "표본 부족"
        excl = lambda q: q[1] > 0 or q[2] < 0
        if o[0] >= T and r[0] >= T and excl(o) and excl(r):
            return "검증됨"
        if r[0] >= T and excl(r) and h[0] > 0 and h[1] > 0 and min(h) >= T / 2:
            return "최근경향"
        if o[0] > 0 and r[0] > 0 and h[0] > 0 and h[1] > 0:
            return "약함(네 구간 같은 방향)"
        return "아님"

    res = {"건수(묶음)": d.groupby(["reg", "g"]).size().unstack().to_dict()}
    for rg in ("상승장", "횡보·전환", "하락장"):
        x = d[d.reg == rg]
        main_c = cmp(x, "A 가뭄", "C 폭발", "x20", 100)
        main_c["판정"] = verdict(main_c, 1.0)
        out = {"주: 20일 초과수익 A − C (%p)": main_c,
               "서술: A − B 20일": cmp(x, "A 가뭄", "B 보통", "x20", 100),
               "서술: 묶음별 평균 3 · 5 · 20일 (%)": {g: [round(float(y[f"x{k}"].mean() * 100), 2) for k in (3, 5, 20)] for g, y in x.groupby("g")},
               "서술: 20일 고가 돌파만 R A − C": cmp(x[x.brk], "A 가뭄", "C 폭발", "Rb", 1)}
        res[rg] = out
    (results_dir() / "volume_drought.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
