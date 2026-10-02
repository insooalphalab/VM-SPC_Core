"""횡보·전환 안에서 지수 20일선으로 상승 초입을 가려 돌파 — 검증이력 9.90 사전 등록 그대로.

  python research/validate_sideways_ma20.py   → results/sideways_ma20.json  (먼저 validate_sideways_breakout.py)
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

from reclassify_recent import RECENT
from validate_sideways_breakout import judge_level
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars


def index_flags() -> pd.DataFrame:
    k = load_bars(LONG_HISTORY, "069500")["close"]
    m20, m60 = k.rolling(20).mean(), k.rolling(60).mean()
    return pd.DataFrame({"S": (k > m20) & (m20 > m20.shift(5)), "S1": k > m20, "S2": k > m60})


def robust(x: pd.DataFrame) -> dict:
    r = x[x.entry_date >= RECENT].R.sort_values(ascending=False)
    trim = float(r.iloc[int(np.ceil(len(r) * 0.01)):].mean())
    yr = x[x.entry_date.dt.year <= 2025].groupby(x.entry_date.dt.year).R.mean()
    return {"상위 1% 뺀 최근 R": round(trim, 3), "플러스인 해": f"{int((yr > 0).sum())}/{len(yr)}",
            "연도별 R": {int(k): round(float(v), 2) for k, v in yr.items()}, "_ok": trim >= 0.10 and (yr > 0).sum() >= 5}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "sideways_breakout_events.csv", parse_dates=["entry_date"])
    d = d[(d.reg == "횡보·전환") & (d.rs >= 0.70)].copy()
    f = index_flags()
    pos = f.index.searchsorted(d.entry_date) - 1                       # 진입 전날(돌파일)
    for k in f:
        d[k] = f[k].to_numpy()[pos]
    res = {}
    for nm, g in {"S 켜짐(주 판정)": d[d.S], "S 꺼짐": d[~d.S], "S1 20일선 위": d[d.S1], "S1 아님": d[~d.S1],
                  "S2 60일선 위": d[d.S2], "S2 아님": d[~d.S2]}.items():
        lv, rb = judge_level(g, rng), robust(g)
        if lv["판정"] != "아님" and not rb.pop("_ok"):
            lv["판정"] = "아님(②·③ 미달)"
        rb.pop("_ok", None)
        res[nm] = {**lv, **rb}
        print(nm, res[nm])
    (results_dir() / "sideways_ma20.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
