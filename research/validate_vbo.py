"""변동성 돌파(래리 윌리엄스, 전일 범위 × K) — 검증이력 9.71 사전 등록 그대로. 박스 돌파(A)는 9.68 사건표.

  python research/validate_vbo.py   → results/vbo_validation.json
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
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars
from validate_compression_regime import regime
from validate_rs_accel import rs_panel

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST, RS_CUT, KS = vb.COST, 70, (0.5, 0.6)


def sim_trend(o, h, l, c, ma20, t, X, stop):
    """V2: t일 장중 X 체결 → 당일 종가가 손절 이하/20일선 아래면 종가 청산, 다음날부터 9.68 규칙. (수익, 보유일)"""
    if t + vbt.HOLD >= len(c):
        return None
    if c[t] <= stop or c[t] < ma20[t]:
        return c[t] / X - 1, 1
    for j in range(t + 1, t + vbt.HOLD):
        if o[j] <= stop:
            return o[j] / X - 1, j - t + 1
        if l[j] <= stop:
            return stop / X - 1, j - t + 1
        if c[j] < ma20[j]:
            return c[j] / X - 1, j - t + 1
    return c[t + vbt.HOLD - 1] / X - 1, vbt.HOLD


def trades(code: str, up: pd.Series, RS: pd.DataFrame) -> list[dict]:
    b = load_bars(LONG_HISTORY, code)
    if b is None or len(b) < 300 or code not in RS.columns:
        return []
    b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    ok = up.reindex(b.index).fillna(False).to_numpy() & (RS[code].reindex(b.index).to_numpy() >= RS_CUT)
    first = max(int(b.index.searchsorted(START)), 60)
    out = []
    for K in KS:
        X = o + K * np.r_[np.nan, (h - l)[:-1]]
        hit = h >= X
        for mode in ("V1 다음날 시가", "V2 추세 청산"):
            busy = -1
            for t in range(first, len(c) - 1):
                if t <= busy or not ok[t - 1] or not hit[t] or np.isnan(atr[t - 1]) or np.isnan(ma20[t]):
                    continue
                x, unit = X[t], atr[t - 1]
                if mode.startswith("V1"):
                    ret, hold = o[t + 1] / x - 1, 1
                else:
                    sim = sim_trend(o, h, l, c, ma20, t, x, x - unit)
                    if sim is None:
                        continue
                    ret, hold = sim
                busy = t + hold - 1
                R = (ret - COST) / max(unit / x, 0.01)
                out.append({"trig": f"K={K} {mode}", "code": code, "entry_date": b.index[t], "ret": ret, "R": R,
                            "win": float(ret - COST > 0), "hold": hold})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    up = regime() == "상승장"
    RS = rs_panel()
    k = pd.DataFrame([x for code in RS.columns for x in trades(code, up, RS)])
    k.to_csv(results_dir() / "vbo_events.csv", index=False)
    a = pd.read_csv(results_dir() / "entry_trigger_events_rs.csv", parse_dates=["entry_date"])
    a = a[a.trig == "A 박스 돌파"]
    d = pd.concat([k, a], ignore_index=True)
    res = {t: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
               "평균 수익": round(float(g.ret.mean()), 4), "보유일": round(float(g.hold.mean()), 1)} for t, g in d.groupby("trig")}
    for t in sorted(k.trig.unique()):
        x = k[k.trig == t]
        tr = vr.diff_ci(x[x.entry_date < SPLIT], a[a.entry_date < SPLIT], rng)
        te = vr.diff_ci(x[x.entry_date >= SPLIT], a[a.entry_date >= SPLIT], rng)
        res[f"{t} − A 박스 R"] = {"맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                 "verdict": "박스보다 낫다" if te[0] >= 0.10 and te[1] > 0 and tr[0] > 0 else
                                            "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "vbo_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
