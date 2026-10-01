"""지수 쉴 때 버틴 종목(RS 다이버전스) 돌파 — 검증이력 9.63 사전 등록 그대로.

  python research/validate_rs_divergence.py   → results/rs_divergence_validation.json
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
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
IDX_WIN, IDX_MAX, RATIO_WIN, LOOKBACK = 5, 0.005, 20, 10
STOP_R = -0.95       # 손절로 끝난 거래(갭 포함) 근사


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    k = load_bars(LONG_HISTORY, "069500")["close"]
    idx_flat = k / k.shift(IDX_WIN) - 1 <= IDX_MAX
    idx_up = k > k.shift(1)
    ev = pd.read_csv(results_dir() / "trend_principles_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    div_any, up_day = [], []
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        ratio = (b["close"] / k.reindex(b.index)).dropna()
        div = (ratio > ratio.rolling(RATIO_WIN).max().shift(1)) & idx_flat.reindex(ratio.index).fillna(False)
        recent = div.astype(float).rolling(LOOKBACK, min_periods=1).max().astype(bool)
        for i, e in g.iterrows():
            j = b.index.searchsorted(e.entry_date) - 1                    # 돌파일
            d = b.index[j]
            div_any.append((i, bool(recent.get(d, False))))
            up_day.append((i, bool(idx_up.get(d, False))))
    ev["div"] = pd.Series(dict(div_any))
    ev["idx_up"] = pd.Series(dict(up_day))
    ev["sig"] = ev["div"] & ev["idx_up"]
    ev["stopped"] = ev.R <= STOP_R
    ev.to_csv(results_dir() / "rs_divergence_events.csv", index=False)

    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "손절 종료": round(float(g.stopped.mean()), 3)}
    res = {"장세별": {}}
    for r, g in ev.groupby("reg"):
        res["장세별"][r] = {"신호": s(g[g.sig]), "나머지": s(g[~g.sig]), "다이버전스만": s(g[g["div"] & ~g.idx_up]),
                           "지수 상승일만": s(g[~g["div"] & g.idx_up]), "둘 다 아님": s(g[~g["div"] & ~g.idx_up])}
    u = ev[ev.reg == "상승장"]
    for name, x in (("전체", u), ("RS ≥ 70 안", u[u.rs >= 0.70])):
        tr = vr.diff_ci(x[(x.entry_date < SPLIT) & x.sig], x[(x.entry_date < SPLIT) & ~x.sig], rng)
        te = vr.diff_ci(x[(x.entry_date >= SPLIT) & x.sig], x[(x.entry_date >= SPLIT) & ~x.sig], rng)
        res[f"상승장 {name} 신호 − 나머지 R"] = {"n_신호": int(x.sig.sum()), "맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                          "verdict": "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
                                                     "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "rs_divergence_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
