"""횡보장 평균 회귀 3후보 — 박스 하단 지지 · 단기 과매도 반등 · 공포 상태 지수 매수 — 검증이력 9.103 사전 등록 그대로.

  python research/validate_sideways_reversion.py   → results/sideways_reversion.json
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

from reclassify_recent import level_ci
from universe_all import validation_codes
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

COST = 0.003
START, SPLIT, HALF, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
BOX, HOLD1 = 30, 20
REGS = ("상승장", "횡보·전환", "하락장")


def sim(o, h, l, c, e, stop, tgt, hold):
    entry = o[e]
    for j in range(e, min(e + hold, len(c))):
        if j > e and o[j] <= stop:
            return o[j] / entry - 1, j - e + 1
        if j > e and o[j] >= tgt:
            return o[j] / entry - 1, j - e + 1
        if l[j] <= stop:
            return stop / entry - 1, j - e + 1
        if h[j] >= tgt:
            return tgt / entry - 1, j - e + 1
    j = min(e + hold, len(c)) - 1
    return c[j] / entry - 1, j - e + 1


def level(x: pd.DataFrame, rng, T: float, T2: float) -> dict:
    per = lambda lo, hi: x[(x.entry_date >= lo) & (x.entry_date < hi)]
    o, r = per(START, SPLIT), per(SPLIT, END)
    oc = level_ci(o, rng) if len(o) >= 30 else (np.nan,) * 3
    rc = level_ci(r, rng) if len(r) >= 30 else (np.nan,) * 3
    h1, h2 = float(per(SPLIT, HALF).R.mean()), float(per(HALF, END).R.mean())
    rec = rc[0] >= T and rc[1] > 0 and min(h1, h2) >= T2
    v = ("검증됨" if oc[0] >= T and oc[1] > 0 else "최근경향") if rec else "아님"
    return {"건수 앞/최근": [len(o), len(r)], "앞 [CI]": [round(q, 3) for q in oc], "최근 [CI]": [round(q, 3) for q in rc],
            "반쪽": [round(h1, 3), round(h2, 3)], "승률": round(float((x.R > 0).mean()), 3), "판정": v}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    idx = load_bars(LONG_HISTORY, "069500")
    bars = {}
    for code in validation_codes():
        b = load_bars(LONG_HISTORY, code)
        if b is not None and len(b) >= 300:
            bars[code] = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    res = {}
    # S1 박스 하단 지지
    rows = []
    for code, b in bars.items():
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        n = len(c)
        H = pd.Series(h).rolling(BOX).max().shift(1).to_numpy()
        L = pd.Series(l).rolling(BOX).min().shift(1).to_numpy()
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        first = max(int(b.index.searchsorted(START)), BOX + 3)
        busy = -1
        for t in range(first, n - 2):
            if t <= busy or np.isnan(H[t]) or np.isnan(atr[t]):
                continue
            w = H[t] / L[t] - 1
            if not (0.10 <= w <= 0.40):
                continue
            if not (l[t] <= L[t] + 0.1 * (H[t] - L[t]) and c[t] > L[t] and c[t] > o[t] and c[t] > c[t - 1]):
                continue
            if any(c[k] < L[k] for k in range(t - 3, t)):
                continue
            e = t + 1
            stop = L[t] - 0.5 * atr[t]
            M = (H[t] + L[t]) / 2
            if not stop < o[e] < M:
                continue
            ret, days = sim(o, h, l, c, e, stop, M, HOLD1)
            retH, _ = sim(o, h, l, c, e, stop, H[t], HOLD1)
            sp = max(1 - stop / o[e], 0.01)
            busy = e + days - 1
            rows.append({"code": code, "entry_date": b.index[e], "reg": reg.get(b.index[t], np.nan),
                         "R": (ret - COST) / sp, "RH": (retH - COST) / sp, "days": days, "tgt": ret > 0})
    s1 = pd.DataFrame(rows)
    s1.to_csv(results_dir() / "sideways_reversion_s1.csv", index=False)
    res["S1 박스 하단 지지(목표 = 중간)"] = {rg: level(s1[s1.reg == rg], rng, 0.20, 0.10) for rg in REGS}
    res["S1 서술: 목표 = 상단"] = {rg: level(s1[s1.reg == rg].assign(R=lambda z: z.RH), rng, 0.20, 0.10) for rg in REGS}
    # S2 단기 과매도 반등
    C = pd.DataFrame({k: b["close"] for k, b in bars.items()}).sort_index()
    C = C[C.index >= START - pd.Timedelta(days=400)]
    r5 = C / C.shift(5) - 1
    up120 = C > C.rolling(120, min_periods=100).mean()
    rk = r5.rank(axis=1, pct=True)
    sig = (rk <= 0.10) & up120
    io, ic = idx["open"], idx["close"]
    rows = []
    for code, b in bars.items():
        s = sig[code].reindex(b.index).fillna(False).to_numpy()
        o, c = b["open"].to_numpy(float), b["close"].to_numpy(float)
        iov, icv = io.reindex(b.index).to_numpy(float), ic.reindex(b.index).to_numpy(float)
        last = -10
        for t in np.flatnonzero(s):
            if b.index[t] < START or t - last < 5 or t + 10 >= len(c):
                continue
            e = t + 1
            if not np.isfinite(iov[e]) or not np.isfinite(icv[t + 10]):
                continue
            last = t
            x5 = (c[t + 5] / o[e] - 1) - (icv[t + 5] / iov[e] - 1) - COST
            x10 = (c[t + 10] / o[e] - 1) - (icv[t + 10] / iov[e] - 1) - COST
            rows.append({"code": code, "entry_date": b.index[e], "reg": reg.get(b.index[t], np.nan), "x5": x5 * 100, "R": x10 * 100})
    s2 = pd.DataFrame(rows)
    s2.to_csv(results_dir() / "sideways_reversion_s2.csv", index=False)
    res["S2 단기 과매도 반등(10일 초과 %p)"] = {rg: level(s2[s2.reg == rg], rng, 1.0, 0.5) for rg in REGS}
    res["S2 서술: 5일 초과 %p"] = {rg: level(s2[s2.reg == rg].assign(R=lambda z: z.x5), rng, 1.0, 0.5) for rg in REGS}
    # S3 공포 상태 지수 매수
    g = pd.read_csv(results_dir() / "regime_gmm_daily.csv", index_col=0, parse_dates=True)
    st = g["gmm"]
    k = idx.reindex(g.index)
    ko, kc = k["open"].to_numpy(float), k["close"].to_numpy(float)
    f20 = pd.Series(np.r_[kc[20:] / ko[1:-19] - 1, [np.nan] * 20], index=g.index)   # t+1 시가 → t+20 종가
    base = {"앞": float(f20[(f20.index >= START) & (f20.index < SPLIT)].mean()), "최근": float(f20[f20.index >= SPLIT].mean())}
    starts = []
    s_arr = st.to_numpy()
    for i in range(20, len(s_arr)):
        if s_arr[i] == 1 and not (s_arr[i - 20:i] == 1).any():
            starts.append(i)
    ep = [{"date": str(g.index[i].date()), "장세": g["ma60"].iloc[i], "20일 수익 %": round(float(f20.iloc[i]) * 100, 1)} for i in starts if np.isfinite(f20.iloc[i])]
    out3 = {"구간": ep}
    ok = True
    for pn, lo, hi in (("앞", START, SPLIT), ("최근", SPLIT, END)):
        e = [x for x in ep if lo <= pd.Timestamp(x["date"]) < hi]
        xs = np.array([x["20일 수익 %"] for x in e]) - base[pn] * 100
        out3[pn] = {"구간 수": len(e), "기준선 %": round(base[pn] * 100, 2), "평균 초과 %p": round(float(xs.mean()), 2) if len(xs) else None,
                    "플러스 구간": f"{int((np.array([x['20일 수익 %'] for x in e]) > 0).sum())}/{len(e)}"}
        need = 5 if pn == "앞" else 3
        ok &= len(xs) >= need and xs.mean() >= 3.0 and (np.array([x["20일 수익 %"] for x in e]) > 0).mean() >= 2 / 3
    out3["판정"] = "검증됨(표본 작음)" if ok else "아님"
    res["S3 공포 상태 지수 매수"] = out3
    (results_dir() / "sideways_reversion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
