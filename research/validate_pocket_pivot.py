"""포켓 피봇(돌파 전 박스 안 진입) — 검증이력 9.59 사전 등록 기준 그대로.

  python research/validate_pocket_pivot.py   → results/pocket_pivot_validation.json
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
import validate_rebreakout as vr
import validate_stops as vs
from box_rules import BOX
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST = vb.COST
HOLD = 60


def sim_managed(o, h, l, c, ma20, e, stop):
    """글의 운용: +2R 절반 익절, 나머지 20일선 이탈 마감. 2R 전 +1R 찍고 종가 ≤ 진입가면 그 종가에 전량. 손절은 장중(갭이면 시가)."""
    if e + HOLD - 1 >= len(c):
        return None
    entry = o[e]
    risk = entry - stop
    t1, t2 = entry + risk, entry + 2 * risk
    half, touched1, done = None, False, 0.0
    for j in range(e, e + HOLD):
        if j > e and o[j] <= stop:
            px = o[j]
            return (0.5 * half + 0.5 * (px / entry - 1)) if half is not None else px / entry - 1
        if l[j] <= stop:
            return (0.5 * half + 0.5 * (stop / entry - 1)) if half is not None else stop / entry - 1
        if half is None:
            if (j > e and o[j] >= t2) or h[j] >= t2:
                half = (o[j] if j > e and o[j] >= t2 else t2) / entry - 1
            else:
                touched1 = touched1 or h[j] >= t1
                if touched1 and c[j] <= entry:
                    return c[j] / entry - 1
        if c[j] < ma20[j]:
            return (0.5 * half + 0.5 * (c[j] / entry - 1)) if half is not None else c[j] / entry - 1
    last = c[e + HOLD - 1] / entry - 1
    return (0.5 * half + 0.5 * last) if half is not None else last


def events(code: str, reg: pd.Series, rng) -> list[dict]:
    bars = load_bars(LONG_HISTORY, code)
    if bars is None or len(bars) < 400:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    ma50 = pd.Series(c).rolling(50).mean().to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    Hs = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    rg = reg.reindex(bars.index).ffill().to_numpy()
    first = max(int(bars.index.searchsorted(START)), 60)
    lo_c, hi_c = first, len(c) - HOLD
    out, busy = [], -1
    for t in range(first, len(c) - 1):
        if t <= busy or np.isnan(ma50[t]) or np.isnan(Hs[t]):
            continue
        if not (c[t] > ma50[t] and c[t] <= Hs[t] and c[t] > c[t - 1] and c[t] > o[t]):
            continue
        rng_t = h[t] - l[t]
        if rng_t <= 0 or (h[t] - c[t]) / rng_t >= 0.3:
            continue
        if not v[t - 10:t].mean() < v[t - 50:t - 10].mean():
            continue
        down = [v[k] for k in range(t - 10, t) if c[k] < c[k - 1]]
        if not down or not v[t] > max(down):
            continue
        e, stop = t + 1, l[t - 9:t + 1].min()
        if not stop < o[e]:
            continue
        trend = vbt.simulate(o, h, l, c, ma20, e, stop)
        if trend is None:
            continue
        managed = sim_managed(o, h, l, c, ma20, e, stop)
        busy = e + trend[2] - 1
        sp = 1 - stop / o[e]
        days = rng.integers(lo_c, hi_c, 10)
        ctrl_t = np.mean([vbt.simulate(o, h, l, c, ma20, int(d), o[d] * (1 - sp))[0] for d in days])
        ctrl_m = np.mean([sim_managed(o, h, l, c, ma20, int(d), o[d] * (1 - sp)) for d in days])
        brk = bool((c[e:e + 30] > Hs[t]).any())
        out.append({"code": code, "entry_date": bars.index[e], "reg": rg[t], "stop_pct": sp, "ret": trend[0], "ctrl": float(ctrl_t),
                    "win": float(trend[0] - COST > 0), "R": (trend[0] - COST) / max(sp, 0.01),
                    "m_ret": managed, "m_ctrl": float(ctrl_m), "m_win": float(managed - COST > 0), "m_R": (managed - COST) / max(sp, 0.01),
                    "broke_out_30d": brk})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    pp = pd.DataFrame([ev for code in codes for ev in events(code, reg, rng)])
    pp.to_csv(results_dir() / "pocket_pivot_events.csv", index=False)
    fb = pd.read_csv(results_dir() / "breakout_edge_events.csv", parse_dates=["entry_date"])
    res = {}
    for r in ("상승장", "횡보·전환", "하락장"):
        g, f = pp[pp.reg == r], fb[fb.reg == r]
        res[r] = {"포켓 피봇(추세 청산)": {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                                        "초과": round(float((g.ret - g.ctrl).mean()), 4), "손절 거리": round(float(g.stop_pct.median()), 4),
                                        "30일 안 돌파": round(float(g.broke_out_30d.mean()), 3)},
                  "포켓 피봇(글의 운용)": {"승률": round(float(g.m_win.mean()), 3), "R": round(float(g.m_R.mean()), 3),
                                        "초과": round(float((g.m_ret - g.m_ctrl).mean()), 4)},
                  "첫 돌파": {"n": int(len(f)), "승률": round(float(f.win.mean()), 3), "R": round(float(f.R.mean()), 3),
                            "초과": round(float((f.ret - f.ctrl).mean()), 4), "손절 거리": round(float(f.stop_pct.median()), 4)}}
    pu, fu = pp[pp.reg == "상승장"], fb[fb.reg == "상승장"]
    tr = vr.diff_ci(pu[pu.entry_date < SPLIT], fu[fu.entry_date < SPLIT], rng)
    te = vr.diff_ci(pu[pu.entry_date >= SPLIT], fu[fu.entry_date >= SPLIT], rng)
    res["상승장 포켓 피봇 − 첫 돌파 R"] = {"맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te]}
    res["verdict"] = "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"
    (results_dir() / "pocket_pivot_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
