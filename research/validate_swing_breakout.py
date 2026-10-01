"""직전 스윙 고점(프랙탈) 돌파 방아쇠 — 검증이력 9.70 사전 등록 그대로. 박스 돌파(A)는 9.68 사건표를 그대로 쓴다.

  python research/validate_swing_breakout.py   → results/swing_breakout_validation.json
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
from box_rules import BOX
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars
from validate_compression_regime import regime
from validate_rs_accel import rs_panel

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST, RS_CUT, SIDE, STOP_R, REACH = vb.COST, 70, 3, -0.95, 20


def swing_level(h: np.ndarray) -> np.ndarray:
    """P[t] = t 종가 시점까지 확정된(오른쪽 3일 지난) 최근 30일 안 가장 최근 스윙 고점의 고가."""
    n = len(h)
    is_sw = np.zeros(n, bool)
    for s in range(SIDE, n - SIDE):
        nb = np.r_[h[s - SIDE:s], h[s + 1:s + SIDE + 1]]
        is_sw[s] = h[s] > nb.max()
    P = np.full(n, np.nan)
    last = -1
    for t in range(n):
        s = t - SIDE                          # t 종가 시점에 오른쪽 3일(s+1..t)이 다 나와 확정되는 봉우리
        if s >= 0 and is_sw[s]:
            last = s
        if last >= 0 and t - last <= BOX:
            P[t] = h[last]
    return P


def trades(code: str, up: pd.Series, RS: pd.DataFrame) -> list[dict]:
    b = load_bars(LONG_HISTORY, code)
    if b is None or len(b) < 300 or code not in RS.columns:
        return []
    b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    P = swing_level(h)
    H = b["high"].rolling(BOX).max().shift(1).to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    ok = up.reindex(b.index).fillna(False).to_numpy() & (RS[code].reindex(b.index).to_numpy() >= RS_CUT)
    prevc = np.r_[np.nan, c[:-1]]
    sig = (c > P) & (prevc <= P) & ok
    first = max(int(b.index.searchsorted(START)), 60)
    out = []
    for mode in ("S 스윙 돌파", "S_early 박스 안 조기"):
        busy = -1
        for t in np.flatnonzero(sig):
            if mode.startswith("S_early") and not c[t] <= H[t]:
                continue
            if t < first or t <= busy or t + 1 >= len(c) or np.isnan(atr[t]) or np.isnan(ma20[t]):
                continue
            e, stop = t + 1, c[t] - atr[t]
            if not stop < o[e]:
                continue
            sim = vbt.simulate(o, h, l, c, ma20, e, stop)
            if sim is None:
                continue
            busy = e + sim[2] - 1
            sp = 1 - stop / o[e]
            R = (sim[0] - COST) / max(sp, 0.01)
            reach = bool((c[t + 1:t + 1 + REACH] > H[t]).any()) if not np.isnan(H[t]) else False
            out.append({"trig": mode, "code": code, "entry_date": b.index[e], "ret": sim[0], "R": R, "win": float(sim[0] - COST > 0),
                        "hold": sim[2], "stop_pct": sp, "stopped": R <= STOP_R, "in_box": bool(c[t] <= H[t]), "reach_H": reach})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    up = regime() == "상승장"
    RS = rs_panel()
    k = pd.DataFrame([x for code in RS.columns for x in trades(code, up, RS)])
    k.to_csv(results_dir() / "swing_breakout_events.csv", index=False)
    a = pd.read_csv(results_dir() / "entry_trigger_events_rs.csv", parse_dates=["entry_date"])
    a = a[a.trig == "A 박스 돌파"]
    d = pd.concat([k, a], ignore_index=True)
    res = {}
    for t, g in d.groupby("trig"):
        res[t] = {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                  "평균 수익": round(float(g.ret.mean()), 4), "손절 거리(중앙)": round(float(g.stop_pct.median()), 4),
                  "−0.95R 이하 종료": round(float(g.stopped.mean()), 3), "보유일": round(float(g.hold.mean()), 1)}
        if t.startswith("S"):
            res[t]["박스 안 신호 비율"] = round(float(g.in_box.mean()), 3)
            res[t]["20일 안 박스 상단 돌파"] = round(float(g.reach_H.mean()), 3)
    for mode in ("S 스윙 돌파", "S_early 박스 안 조기"):
        x = k[k.trig == mode]
        tr = vr.diff_ci(x[x.entry_date < SPLIT], a[a.entry_date < SPLIT], rng)
        te = vr.diff_ci(x[x.entry_date >= SPLIT], a[a.entry_date >= SPLIT], rng)
        res[f"{mode.split()[0]} − A 박스 R"] = {"맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                              "verdict": "박스보다 낫다" if te[0] >= 0.10 and te[1] > 0 and tr[0] > 0 else
                                                         "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "swing_breakout_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
