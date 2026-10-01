"""돌파 손절을 박스 상단으로 — 검증이력 9.57 사전 등록 기준 그대로.

  python research/validate_breakout_stop_h.py   → results/breakout_stop_h_validation.json
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
import validate_stops as vs
from box_rules import BOX
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST = vb.COST
ENTRIES = (0, 5, 10)          # 0 = 첫 돌파 다음 날, 5·10 = 그만큼 유지 확인 후
HOLD = vbt.HOLD


def sim(o, h, l, c, ma20, e, stop, mode):
    """mode: 'atr'·'intraday' = 장중 손절(갭이면 시가), 'close' = 종가 < stop 이면 그 종가. 20일선 이탈 마감·HOLD일."""
    if e + HOLD - 1 >= len(c):
        return None
    entry = o[e]
    for j in range(e, e + HOLD):
        if mode != "close":
            if j > e and o[j] <= stop:
                return o[j] / entry - 1, j - e + 1
            if l[j] <= stop:
                return stop / entry - 1, j - e + 1
        elif c[j] < stop:
            return c[j] / entry - 1, j - e + 1
        if c[j] < ma20[j]:
            return c[j] / entry - 1, j - e + 1
    return c[e + HOLD - 1] / entry - 1, HOLD


def events(code: str, reg: pd.Series, rng) -> list[dict]:
    bars = load_bars(LONG_HISTORY, code)
    if bars is None or len(bars) < 400:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    Hs = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    rg = reg.reindex(bars.index).ffill().to_numpy()
    first = max(int(bars.index.searchsorted(START)), 60)
    lo_c, hi_c = first, len(c) - HOLD
    out = []
    for n in ENTRIES:
        busy = -1
        for t1 in range(first, len(c) - n - 2):
            if np.isnan(Hs[t1]) or not (c[t1] > Hs[t1] and c[t1 - 1] <= Hs[t1 - 1]):
                continue
            H = Hs[t1]
            if n and (c[t1 + 1:t1 + n + 1] < H).any():
                continue
            tc = t1 + n
            if tc <= busy:
                continue
            e = tc + 1
            stop_atr = H - atr[tc]
            if not stop_atr < o[e]:
                continue
            base = sim(o, h, l, c, ma20, e, stop_atr, "atr")
            if base is None:
                continue
            busy = e + base[1] - 1
            row = {"code": code, "entry": n, "entry_date": bars.index[e], "reg": rg[tc]}
            days = rng.integers(lo_c, hi_c, 10)
            for mode, stop in (("현행 H−ATR", stop_atr), ("장중 H", H), ("종가 H", H)):
                m = "atr" if mode == "현행 H−ATR" else "intraday" if mode == "장중 H" else "close"
                r = sim(o, h, l, c, ma20, e, stop, m)
                risk = max(1 - stop / o[e], 0.005)
                ctrl = np.mean([sim(o, h, l, c, ma20, int(d), o[d] * (1 - risk), m)[0] for d in days])
                row[f"{mode}|R"] = (r[0] - COST) / risk
                row[f"{mode}|win"] = float(r[0] - COST > 0)
                row[f"{mode}|ex"] = r[0] - ctrl
                row[f"{mode}|risk"] = risk
            out.append(row)
    return out


def diff_ci(g: pd.DataFrame, a: str, b: str, rng) -> tuple[float, float, float]:
    x = g.assign(dd=g[a] - g[b])
    blk = (x.entry_date.rank(method="dense").astype(int) // vb.BLOCK).to_numpy()
    lo, hi = vb.boot_mean(x["dd"].to_numpy(), blk, rng)
    return float(x["dd"].mean()), lo, hi


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    d = pd.DataFrame([ev for code in codes for ev in events(code, reg, rng)])
    d.to_csv(results_dir() / "breakout_stop_h_events.csv", index=False)
    res = {}
    for n in ENTRIES:
        lab = "첫 돌파 다음 날" if n == 0 else f"{n}일 유지 확인 후"
        g = d[(d.entry == n) & (d.reg == "상승장")]
        r = {"n(상승장)": int(len(g))}
        for mode in ("현행 H−ATR", "장중 H", "종가 H"):
            r[mode] = {"승률": round(float(g[f"{mode}|win"].mean()), 3), "R": round(float(g[f"{mode}|R"].mean()), 3),
                       "초과": round(float(g[f"{mode}|ex"].mean()), 4), "손절 거리": round(float(g[f"{mode}|risk"].median()), 4)}
        for mode in ("장중 H", "종가 H"):
            tr = diff_ci(g[g.entry_date < SPLIT], f"{mode}|R", "현행 H−ATR|R", rng)
            te = diff_ci(g[g.entry_date >= SPLIT], f"{mode}|R", "현행 H−ATR|R", rng)
            r[f"{mode} − 현행 R"] = {"맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te],
                                    "verdict": "채택" if te[0] >= 0.10 and te[1] > 0 and tr[0] > 0 else
                                               "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
        res[lab] = r
    (results_dir() / "breakout_stop_h_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
