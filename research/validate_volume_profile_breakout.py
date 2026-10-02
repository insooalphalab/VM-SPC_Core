"""6개월 매물대 상단 돌파 — 상단을 두드린 뒤 거래량 돌파 · 상단 +10~20% · 다음날 돌파일 시가 지킴 — 검증이력 9.108 사전 등록 그대로.

  python research/validate_volume_profile_breakout.py   → results/volume_profile_breakout.json
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

from build_winrate_table import pf
from reclassify_recent import level_ci
from universe_all import validation_codes
from validate_breakout_exits import EXITS, simulate
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

COST = 0.003
START, SPLIT, HALF, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
WIN, BINS, VA = 120, 30, 0.70
REGS = ("상승장", "횡보·전환", "하락장")


def value_area_high(h, l, c, v) -> float:
    tp = (h + l + c) / 3
    lo, hi = l.min(), h.max()
    if hi <= lo:
        return np.nan
    edges = np.linspace(lo, hi, BINS + 1)
    vol = np.bincount(np.clip(np.searchsorted(edges, tp, side="right") - 1, 0, BINS - 1), weights=v, minlength=BINS)
    a = b = int(np.argmax(vol))
    tot, acc = vol.sum(), vol[a]
    while acc < VA * tot and (a > 0 or b < BINS - 1):
        up = vol[b + 1] if b < BINS - 1 else -1
        dn = vol[a - 1] if a > 0 else -1
        if up >= dn:
            b += 1
            acc += vol[b]
        else:
            a -= 1
            acc += vol[a]
    return float(edges[b + 1])


def margin_bin(m: float) -> str:
    return "0~10%" if m < 0.10 else "10~20%" if m < 0.20 else "20%↑"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    rows = []
    for code in validation_codes():
        b = load_bars(LONG_HISTORY, code)
        if b is None or len(b) < 400:
            continue
        b = b[(b[["open", "high", "low", "close", "volume"]] > 0).all(1)]
        o, h, l, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        n = len(c)
        va20 = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        ma = pd.Series(c).rolling(50).mean().to_numpy()
        first = max(int(b.index.searchsorted(START)), WIN + 1)
        busy = -1
        for t in range(first, n - 3):
            if t <= busy or not (v[t] >= 2 * va20[t] and c[t] > o[t]):
                continue
            w = slice(t - WIN, t)
            if h[w].max() / l[w].min() - 1 > 0.50:
                continue
            V = value_area_high(h[w], l[w], c[w], v[w])
            if not np.isfinite(V) or c[t] <= V:
                continue
            s20 = slice(t - 20, t)
            if (h[s20] >= 0.98 * V).sum() < 2 or (c[s20] > V).any():
                continue
            m = c[t] / V - 1
            hold = bool(c[t + 1] >= o[t])
            e, stop = t + 2, o[t]
            row = {"code": code, "date": b.index[t], "entry_date": b.index[e], "reg": reg.get(b.index[t], np.nan),
                   "margin": m, "mbin": margin_bin(m), "hold": hold, "R": np.nan}
            if o[e] > stop:
                ret, days = simulate(o, h, l, c, ma, atr, e, stop, EXITS["E2 50일선"])
                row["R"] = (ret - COST) / max(1 - stop / o[e], 0.01)
                busy = e + days - 1
            rows.append(row)
    d = pd.DataFrame(rows).dropna(subset=["R"])
    d.to_csv(results_dir() / "volume_profile_breakout_events.csv", index=False)

    def level(x: pd.DataFrame) -> dict:
        per = lambda lo, hi: x[(x.entry_date >= lo) & (x.entry_date < hi)]
        o_, r_ = per(START, SPLIT), per(SPLIT, END)
        oc = level_ci(o_, rng) if len(o_) >= 30 else (np.nan,) * 3
        rc = level_ci(r_, rng) if len(r_) >= 30 else (np.nan,) * 3
        h1, h2 = float(per(SPLIT, HALF).R.mean()), float(per(HALF, END).R.mean())
        if len(o_) < 30 or len(r_) < 30:
            v = "표본 부족"
        elif rc[0] >= 0.20 and rc[1] > 0:
            v = "검증됨" if oc[0] >= 0.20 and oc[1] > 0 else ("최근경향" if min(h1, h2) >= 0.10 else "아님")
        else:
            v = "아님"
        return {"건수 앞/최근": [len(o_), len(r_)], "앞 R [CI]": [round(q, 2) for q in oc], "최근 R [CI]": [round(q, 2) for q in rc],
                "반쪽": [round(h1, 2), round(h2, 2)], "승률": round(float((x.R > 0).mean()), 3), "R": round(float(x.R.mean()), 2),
                "수익 배수": pf(x.R), "판정": v}

    res = {"전체 사건": len(d)}
    for rg in REGS:
        x = d[d.reg == rg]
        out = {"주: 마진 10~20% + 다음날 시가 지킴": level(x[(x.mbin == "10~20%") & x.hold])}
        for mb in ("0~10%", "20%↑"):
            out[f"서술: 마진 {mb} + 다음날 지킴"] = level(x[(x.mbin == mb) & x.hold])
        out["서술: 마진 10~20% · 다음날 못 지킴"] = level(x[(x.mbin == "10~20%") & ~x.hold])
        out["서술: 매물대 돌파 전체"] = level(x)
        res[rg] = out
    (results_dir() / "volume_profile_breakout.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for rg in REGS:
        for k, v in res[rg].items():
            print(rg, k, json.dumps(v, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
