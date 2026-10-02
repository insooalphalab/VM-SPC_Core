"""6개월 매물대를 오르내리다 처음으로 상단 +3% 위 종가 — 검증이력 9.109 사전 등록 그대로.

  python research/validate_volume_profile_first3.py   → results/volume_profile_first3.json
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

import validate_rebreakout as vr
from build_winrate_table import pf
from reclassify_recent import level_ci
from universe_all import validation_codes
from validate_breakout_exits import EXITS, simulate
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

COST = 0.003
START, SPLIT, HALF, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
WIN, BINS, VA, MARGIN, CROSS = 120, 30, 0.70, 0.03, 4
REGS = ("상승장", "횡보·전환", "하락장")


def profile(h, l, c, v) -> tuple[float, float]:
    """(매물대 상단 V, 중심 P)."""
    tp = (h + l + c) / 3
    lo, hi = l.min(), h.max()
    if hi <= lo:
        return np.nan, np.nan
    edges = np.linspace(lo, hi, BINS + 1)
    vol = np.bincount(np.clip(np.searchsorted(edges, tp, side="right") - 1, 0, BINS - 1), weights=v, minlength=BINS)
    a = b = poc = int(np.argmax(vol))
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
    return float(edges[b + 1]), float((edges[poc] + edges[poc + 1]) / 2)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    bars = {}
    for code in validation_codes():
        b = load_bars(LONG_HISTORY, code)
        if b is not None and len(b) >= 400:
            bars[code] = b[(b[["open", "high", "low", "close", "volume"]] > 0).all(1)]
    C = pd.DataFrame({k: b["close"] for k, b in bars.items()}).sort_index()
    r = lambda k: C / C.shift(k) - 1
    RS = (0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)).rank(axis=1, pct=True)
    rows = []
    for code, b in bars.items():
        o, h, l, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        n = len(c)
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        ma = pd.Series(c).rolling(50).mean().to_numpy()
        rs = RS[code].reindex(b.index).to_numpy()
        first = max(int(b.index.searchsorted(START)), WIN + 1)
        for t in range(first, n - 2):
            w = slice(t - WIN, t)
            if h[w].max() / l[w].min() - 1 > 0.50:
                continue
            V, P = profile(h[w], l[w], c[w], v[w])
            if not np.isfinite(V) or c[t] < (1 + MARGIN) * V or (c[w] >= (1 + MARGIN) * V).any():
                continue
            side = np.sign(c[w] - P)
            side = side[side != 0]
            if (np.diff(side) != 0).sum() < CROSS:
                continue
            e, stop = t + 1, c[t] - atr[t]
            if not o[e] > stop or np.isnan(atr[t]):
                continue
            ret, _ = simulate(o, h, l, c, ma, atr, e, stop, EXITS["E2 50일선"])
            rows.append({"code": code, "entry_date": b.index[e], "reg": reg.get(b.index[t], np.nan), "rs": rs[t],
                         "margin": c[t] / V - 1, "R": (ret - COST) / max(1 - stop / o[e], 0.01)})
    d = pd.DataFrame(rows)
    d.to_csv(results_dir() / "volume_profile_first3_events.csv", index=False)
    base = pd.read_csv(results_dir() / "sideways_breakout_events.csv", parse_dates=["entry_date"])
    base = base[(base.reg == "상승장") & (base.rs >= 0.70)].assign(R=lambda z: z.R2)

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

    res = {"전체 신호": len(d)}
    for rg in REGS:
        x = d[d.reg == rg]
        res[rg] = {"주: 처음 +3% 마감": level(x), "서술: RS ≥ 70만": level(x[x.rs >= 0.70])}
    up = d[(d.reg == "상승장") & (d.rs >= 0.70)]
    cmp = {}
    for pn, lo, hi in (("앞", START, SPLIT), ("최근", SPLIT, END)):
        a = up[(up.entry_date >= lo) & (up.entry_date < hi)]
        bb = base[(base.entry_date >= lo) & (base.entry_date < hi)]
        cmp[pn] = [round(q, 3) for q in vr.diff_ci(a, bb, rng)] if len(a) >= 30 else None
    ok = cmp["앞"] and cmp["최근"] and cmp["최근"][0] >= 0.15 and np.sign(cmp["앞"][0]) == np.sign(cmp["최근"][0])
    res["상승장 RS ≥ 70: 매물대 첫 +3% − 현행 30일 박스 돌파 R [CI]"] = {**cmp, "현행 R": round(float(base.R.mean()), 2),
                                                                "판정": "추가 후보" if ok else "아님"}
    (results_dir() / "volume_profile_first3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
