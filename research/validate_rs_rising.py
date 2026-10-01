"""RS 70 미만이지만 치고 올라오는 종목(RS 레벨 + 변화량) — 검증이력 9.65 사전 등록 그대로.

  python research/validate_rs_rising.py   → results/rs_rising_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import itertools
import json
import sys
import warnings

import numpy as np
import pandas as pd

import validate_rebreakout as vr
from validate_rs_accel import rs_panel
from v2_config import results_dir

SPLIT = pd.Timestamp("2021-06-01")
RS_CUT, MIN_FIT = 70, 300
LEVELS, LAGS, DELTAS = (40, 50, 60), (10, 20, 40, 60), (10, 15, 20)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    RS = rs_panel()
    ev = pd.read_csv(results_dir() / "trend_principles_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    ev = ev[(ev.reg == "상승장") & ev.code.isin(RS.columns)].copy()
    pos = RS.index.searchsorted(ev.entry_date) - 1
    ok = pos >= max(LAGS)
    ev, pos = ev[ok].copy(), pos[ok]
    col = RS.columns.get_indexer(ev.code)
    arr = RS.to_numpy()
    ev["rs_now"] = arr[pos, col]
    for lag in LAGS:
        ev[f"d{lag}"] = ev.rs_now - arr[pos - lag, col]
    ev = ev.dropna(subset=["rs_now"] + [f"d{lag}" for lag in LAGS])
    fit, test = ev.entry_date < SPLIT, ev.entry_date >= SPLIT
    top = ev.rs_now >= RS_CUT
    grid = []
    for L, lag, D in itertools.product(LEVELS, LAGS, DELTAS):
        add = (ev.rs_now >= L) & ~top & (ev[f"d{lag}"] >= D)
        rest = ~top & ~add
        row = {"L": L, "lag": lag, "D": D}
        for nm, m in (("맞춤", fit), ("예측", test)):
            row[f"{nm} 추가 n"] = int((add & m).sum())
            row[f"{nm} 추가 R"] = round(float(ev.R[add & m].mean()), 3)
            row[f"{nm} 제외 R"] = round(float(ev.R[rest & m].mean()), 3)
            row[f"{nm} 차이"] = round(row[f"{nm} 추가 R"] - row[f"{nm} 제외 R"], 3)
        grid.append(row)
    g = pd.DataFrame(grid)
    pick = g[g["맞춤 추가 n"] >= MIN_FIT].sort_values("맞춤 차이", ascending=False).iloc[0]
    L, lag, D = int(pick.L), int(pick.lag), int(pick.D)
    add = (ev.rs_now >= L) & ~top & (ev[f"d{lag}"] >= D)
    rest = ~top & ~add
    te = vr.diff_ci(ev[add & test], ev[rest & test], rng)
    tr = vr.diff_ci(ev[add & fit], ev[rest & fit], rng)
    top_te = float(ev.R[top & test].mean())
    add_te = float(ev.R[add & test].mean())
    first = te[0] >= 0.15 and te[1] > 0
    s = lambda x: {"n": int(len(x)), "승률": round(float(x.win.mean()), 3), "R": round(float(x.R.mean()), 3),
                   "손절 종료": round(float((x.R <= -0.95).mean()), 3)}
    res = {"고른 조합": {"L": L, "lag": lag, "D": D},
           "예측 기간": {"RS≥70": s(ev[top & test]), "추가": s(ev[add & test]), "제외": s(ev[rest & test])},
           "맞춤 기간": {"RS≥70": s(ev[top & fit]), "추가": s(ev[add & fit]), "제외": s(ev[rest & fit])},
           "추가 − 제외 R": {"맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te]},
           "추가 R − RS≥70 R(예측)": round(add_te - top_te, 3),
           "verdict": "채택" if first and add_te >= top_te - 0.10 else "방향만 일치" if first or (te[0] > 0 and tr[0] > 0) else "채택 안 함",
           "격자 예측 기간 추가 > 제외 비율": round(float((g["예측 차이"] > 0).mean()), 3),
           "격자": grid}
    (results_dir() / "rs_rising_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "격자"}, ensure_ascii=False, indent=1))
    print(g.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
