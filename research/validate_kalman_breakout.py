"""칼만 상한(+2σ) 돌파 방아쇠 — 검증이력 9.69 사전 등록 그대로. 박스 돌파(A)는 9.68 사건표를 그대로 쓴다.

  python research/validate_kalman_breakout.py   → results/kalman_breakout_validation.json
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
from v2_config import LONG_HISTORY, Params, results_dir
from v2_datastore import load_bars
from v2_signals import kalman_features
from validate_compression_regime import regime
from validate_rs_accel import rs_panel

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST, RS_CUT, BAND, STOP_R = vb.COST, 70, 2.0, -0.95


def sim_center(o, c, center, e) -> tuple[float, int] | None:
    """K2: 다음날 시가 진입 → 종가 < 칼만 중심선이면 그날 종가 청산, 최대 HOLD일."""
    if e + vbt.HOLD - 1 >= len(c):
        return None
    for j in range(e, e + vbt.HOLD):
        if c[j] < center[j]:
            return c[j] / o[e] - 1, j - e + 1
    return c[e + vbt.HOLD - 1] / o[e] - 1, vbt.HOLD


def trades(code: str, up: pd.Series, RS: pd.DataFrame) -> list[dict]:
    b = load_bars(LONG_HISTORY, code)
    if b is None or len(b) < 300 or code not in RS.columns:
        return []
    b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    f = kalman_features(b, Params())
    z = f["band_z"].to_numpy()
    center = np.exp(f["level"].to_numpy())
    H = b["high"].rolling(BOX).max().shift(1).to_numpy()
    box = (c > H) & (np.r_[np.nan, c[:-1]] <= np.r_[np.nan, H[:-1]])
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    ok = up.reindex(b.index).fillna(False).to_numpy() & (RS[code].reindex(b.index).to_numpy() >= RS_CUT)
    sig = (z > BAND) & (np.r_[np.nan, z[:-1]] <= BAND) & ok
    first = max(int(b.index.searchsorted(START)), 60)
    out = []
    for mode in ("K1 칼만 · 공통 규칙", "K2 칼만 · 중심선 청산"):
        busy = -1
        for t in np.flatnonzero(sig):
            if t < first or t <= busy or t + 1 >= len(c) or np.isnan(atr[t]) or np.isnan(ma20[t]) or np.isnan(center[t]):
                continue
            e = t + 1
            if mode.startswith("K1"):
                stop = c[t] - atr[t]
                if not stop < o[e]:
                    continue
                sim = vbt.simulate(o, h, l, c, ma20, e, stop)
                if sim is None:
                    continue
                ret, hold, sp = sim[0], sim[2], 1 - stop / o[e]
            else:
                sim = sim_center(o, c, center, e)
                if sim is None:
                    continue
                ret, hold, sp = sim[0], sim[1], 1 - center[t] / o[e]
            busy = e + hold - 1
            R = (ret - COST) / max(sp, 0.01)
            near_box = bool(box[max(0, t - 2):t + 3].any())
            out.append({"trig": mode, "code": code, "entry_date": b.index[e], "ret": ret, "R": R, "win": float(ret - COST > 0),
                        "hold": hold, "stop_pct": sp, "stopped": R <= STOP_R, "near_box": near_box})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    up = regime() == "상승장"
    RS = rs_panel()
    k = pd.DataFrame([x for code in RS.columns for x in trades(code, up, RS)])
    k.to_csv(results_dir() / "kalman_breakout_events.csv", index=False)
    a = pd.read_csv(results_dir() / "entry_trigger_events_rs.csv", parse_dates=["entry_date"])
    a = a[a.trig == "A 박스 돌파"]
    d = pd.concat([k, a], ignore_index=True)
    res = {t: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
               "평균 수익": round(float(g.ret.mean()), 4), "손절 거리(중앙)": round(float(g.stop_pct.median()), 4),
               "−0.95R 이하 종료": round(float(g.stopped.mean()), 3), "보유일": round(float(g.hold.mean()), 1),
               **({"박스 돌파 ±2일 겹침": round(float(g.near_box.mean()), 3)} if "near_box" in g and g.near_box.notna().all() else {})}
           for t, g in d.groupby("trig")}
    for mode in ("K1 칼만 · 공통 규칙", "K2 칼만 · 중심선 청산"):
        x = k[k.trig == mode]
        tr = vr.diff_ci(x[x.entry_date < SPLIT], a[a.entry_date < SPLIT], rng)
        te = vr.diff_ci(x[x.entry_date >= SPLIT], a[a.entry_date >= SPLIT], rng)
        res[f"{mode[:2]} − A 박스 R"] = {"맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                       "verdict": "박스보다 낫다" if te[0] >= 0.10 and te[1] > 0 and tr[0] > 0 else
                                                  "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "kalman_breakout_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
