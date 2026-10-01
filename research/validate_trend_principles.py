"""추세추종 원칙 4가지(RS 등급 · 와인스타인 2단계 · 52주 신저가 비율 · 신고가 윗꼬리 음봉 청산) — 검증이력 9.62 사전 등록 그대로.

  python research/validate_trend_principles.py   → results/trend_principles_validation.json
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
import validate_breakout_trend as vbt
import validate_rebreakout as vr
from universe_all import validation_codes
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
COST = vb.COST
RS_CUT, MA_LONG, SLOPE, LOW_WIN = 0.70, 150, 20, 250
WICK, VOL_X = 0.5, 1.5


def bars_of(code: str):
    b = load_bars(LONG_HISTORY, code)
    if b is None or len(b) < 300:
        return None
    return b[(b[["open", "high", "low", "close"]] > 0).all(1)]


def simulate_wick(o, h, l, c, v, ma20, vavg, e, stop):
    """기존 규칙 + 신고가 윗꼬리 음봉(거래량) 다음날 시가 청산."""
    if e + vbt.HOLD - 1 >= len(c):
        return None
    entry, peak = o[e], 0.0
    for j in range(e, e + vbt.HOLD):
        if j > e and o[j] <= stop:
            return o[j] / entry - 1
        if l[j] <= stop:
            return stop / entry - 1
        if c[j] < ma20[j]:
            return c[j] / entry - 1
        rng_j = h[j] - l[j]
        new_high = h[j] >= peak
        peak = max(peak, h[j])
        if (new_high and c[j] < o[j] and rng_j > 0 and (h[j] - o[j]) / rng_j >= WICK
                and v[j] >= VOL_X * vavg[j] and j + 1 < len(c) and j + 1 < e + vbt.HOLD):
            return o[j + 1] / entry - 1
    return c[e + vbt.HOLD - 1] / entry - 1


def paired_ci(d: pd.DataFrame, rng) -> tuple[float, float, float]:
    x = d.assign(block=d.entry_date.rank(method="dense").astype(int) // vb.BLOCK)
    agg = x.groupby("block").agg(s=("dR", "sum"), n=("dR", "size")).to_numpy(float)
    idx = rng.integers(0, len(agg), (vb.N_BOOT, len(agg)))
    s = agg[idx].sum(1)
    lo, hi = np.percentile(s[:, 0] / s[:, 1], [2.5, 97.5])
    return float(d.dR.mean()), float(lo), float(hi)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "breakout_edge_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    uni = validation_codes()
    rs, newlow, bars = {}, {}, {}
    for code in sorted(set(uni) | set(ev.code)):
        b = bars_of(code)
        if b is None:
            continue
        bars[code] = b
        if code in uni:
            c = b["close"]
            rs[code] = 0.4 * (c / c.shift(63) - 1) + 0.2 * (c / c.shift(126) - 1) + 0.2 * (c / c.shift(189) - 1) + 0.2 * (c / c.shift(252) - 1)
            lo = c.rolling(LOW_WIN).min().shift(1)
            newlow[code] = (c <= lo).astype(float).where(lo.notna())
    RS = pd.DataFrame(rs).sort_index().rank(axis=1, pct=True)
    NL = pd.DataFrame(newlow).sort_index()
    cnt = NL.notna().sum(1)
    share = (NL.sum(1) / cnt.replace(0, np.nan)).where(cnt >= 100)
    sm = share.rolling(10).mean()
    rising = sm > sm.shift(SLOPE)

    rows = []
    for _, e in ev.iterrows():
        b = bars.get(e.code)
        if b is None:
            continue
        ie = b.index.searchsorted(e.entry_date)
        if ie >= len(b) or b.index[ie] != e.entry_date or ie < MA_LONG + SLOPE:
            continue
        t, d = ie - 1, b.index[ie - 1]
        c = b["close"].to_numpy(float)
        ma = pd.Series(c[: t + 1]).rolling(MA_LONG).mean().to_numpy()
        stage2 = bool(c[t] > ma[t] and ma[t] > ma[t - SLOPE])
        r_rs = RS.at[d, e.code] if (e.code in RS.columns and d in RS.index) else np.nan
        r_br = rising.get(d, np.nan) if pd.notna(sm.get(d, np.nan)) and pd.notna(sm.shift(SLOPE).get(d, np.nan)) else np.nan
        rows.append({**e.to_dict(), "t": t, "rs": r_rs, "stage2": stage2, "breadth_up": r_br})
    d = pd.DataFrame(rows)

    # ④ 청산 규칙 — 같은 사건 다시 돌리기
    wick_R = []
    for code, g in d.groupby("code"):
        b = bars[code]
        o, h, l, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        ma20 = pd.Series(c).rolling(20).mean().to_numpy()
        vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        for i, e in g.iterrows():
            ie = int(e.t) + 1
            r = simulate_wick(o, h, l, c, v, ma20, vavg, ie, o[ie] * (1 - e.stop_pct))
            wick_R.append((i, np.nan if r is None else (r - COST) / max(e.stop_pct, 0.01), np.nan if r is None else r))
    w = pd.DataFrame(wick_R, columns=["i", "R_wick", "ret_wick"]).set_index("i")
    d = d.join(w)
    d.to_csv(results_dir() / "trend_principles_events.csv", index=False)

    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "초과": round(float((g.ret - g.ctrl).mean()), 4)}
    flags = {"① RS ≥ 70": ("rs_hi", "rs"), "② 2단계": ("stage2", "stage2"), "③ 신저가 비율 증가 아님": ("calm", "breadth_up")}
    d["rs_hi"] = d.rs >= RS_CUT
    d["calm"] = d.breadth_up == False  # noqa: E712
    res = {"n": int(len(d)), "장세별": {}}
    for r, g in d.groupby("reg"):
        x = {}
        for name, (col, src) in flags.items():
            gg = g[g[src].notna()]
            x[name] = {"해당": s(gg[gg[col].astype(bool)]), "나머지": s(gg[~gg[col].astype(bool)])}
        gw = g[g.R_wick.notna()]
        x["④ 윗꼬리 음봉 청산"] = {"n": int(len(gw)), "기존 R": round(float(gw.R.mean()), 3), "새 R": round(float(gw.R_wick.mean()), 3),
                               "청산 바뀐 비율": round(float((~np.isclose(gw.ret_wick, gw.ret)).mean()), 3)}
        res["장세별"][r] = x
    u = d[d.reg == "상승장"]
    for name, (col, src) in flags.items():
        a = u[u[src].notna()].copy()
        a[col] = a[col].astype(bool)
        tr = vr.diff_ci(a[(a.entry_date < SPLIT) & a[col]], a[(a.entry_date < SPLIT) & ~a[col]], rng)
        te = vr.diff_ci(a[(a.entry_date >= SPLIT) & a[col]], a[(a.entry_date >= SPLIT) & ~a[col]], rng)
        res[f"상승장 {name} − 나머지 R"] = {"맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te],
                                       "verdict": "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
                                                  "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    a = u[u.R_wick.notna()].assign(dR=lambda x: x.R_wick - x.R)
    tr, te = paired_ci(a[a.entry_date < SPLIT], rng), paired_ci(a[a.entry_date >= SPLIT], rng)
    res["상승장 ④ 새 청산 − 기존 R"] = {"맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te],
                                   "verdict": "채택" if te[0] >= 0.05 and te[1] > 0 and tr[0] > 0 else
                                              "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "trend_principles_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
