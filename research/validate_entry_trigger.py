"""진입 방아쇠 비교 — 박스 돌파 vs 20일선 양봉 돌파 vs 무작위 진입 — 검증이력 9.68 사전 등록 그대로.

  python research/validate_entry_trigger.py   → results/entry_trigger_validation.json
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
from validate_compression_regime import regime
from validate_rs_accel import rs_panel
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST, RS_CUT, P_RANDOM, STOP_R = vb.COST, 70, 0.10, -0.95
TRIGGERS = ("A 박스 돌파", "B 20일선 양봉", "C 무작위")


def trades(code: str, up: pd.Series, RS: pd.DataFrame, need_rs: bool, rng) -> list[dict]:
    b = load_bars(LONG_HISTORY, code)
    if b is None or len(b) < 300 or code not in RS.columns:
        return []
    b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    H = b["high"].rolling(BOX).max().shift(1).to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    ok = up.reindex(b.index).fillna(False).to_numpy()
    rs = RS[code].reindex(b.index).to_numpy()
    if need_rs:
        ok = ok & (rs >= RS_CUT)
    sig = {"A 박스 돌파": (c > H) & (np.r_[np.nan, c[:-1]] <= np.r_[np.nan, H[:-1]]),
           "B 20일선 양봉": (c > ma20) & (np.r_[np.nan, c[:-1]] <= np.r_[np.nan, ma20[:-1]]) & (c > o),
           "C 무작위": rng.random(len(c)) < P_RANDOM}
    first = max(int(b.index.searchsorted(START)), 60)
    out = []
    for name, s in sig.items():
        busy = -1
        for t in np.flatnonzero(s & ok):
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
            out.append({"trig": name, "code": code, "entry_date": b.index[e], "ret": sim[0], "R": R, "win": float(sim[0] - COST > 0),
                        "hold": sim[2], "stop_pct": sp, "stopped": R <= STOP_R})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    up = regime() == "상승장"
    RS = rs_panel()
    res = {}
    for scope, need in (("상승장 · RS ≥ 70", True), ("상승장 전체", False)):
        d = pd.DataFrame([x for code in RS.columns for x in trades(code, up, RS, need, rng)])
        d.to_csv(results_dir() / f"entry_trigger_events_{'rs' if need else 'all'}.csv", index=False)
        r = {k: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                 "평균 수익": round(float(g.ret.mean()), 4), "손절 거리(중앙)": round(float(g.stop_pct.median()), 4),
                 "손절 종료": round(float(g.stopped.mean()), 3), "보유일": round(float(g.hold.mean()), 1)} for k, g in d.groupby("trig")}
        for x, y in (("A 박스 돌파", "C 무작위"), ("B 20일선 양봉", "C 무작위"), ("B 20일선 양봉", "A 박스 돌파")):
            a, b = d[d.trig == x], d[d.trig == y]
            tr = vr.diff_ci(a[a.entry_date < SPLIT], b[b.entry_date < SPLIT], rng)
            te = vr.diff_ci(a[a.entry_date >= SPLIT], b[b.entry_date >= SPLIT], rng)
            r[f"{x[0]} − {y[0]} R"] = {"맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                      "verdict": "더 낫다" if te[0] >= 0.10 and te[1] > 0 and tr[0] > 0 else
                                                 "방향만 일치" if te[0] > 0 and tr[0] > 0 else "차이 없음/반대"}
        res[scope] = r
    (results_dir() / "entry_trigger_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
