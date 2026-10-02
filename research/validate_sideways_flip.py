"""횡보장 돌파를 상승장 전환일까지 살아 있으면 그때 산다 — 검증이력 9.91 사전 등록 그대로.

  python research/validate_sideways_flip.py   → results/sideways_flip.json  (먼저 validate_sideways_breakout.py)
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
from reclassify_recent import RECENT
from validate_breakout_exits import EXITS, simulate
from validate_compression_regime import regime
from validate_sideways_breakout import judge_level
from validate_sideways_ma20 import robust
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

WAIT = 60


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "sideways_breakout_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    d = d[(d.reg == "횡보·전환") & (d.rs >= 0.70)].copy()
    d["code"] = d["code"].str.zfill(6)
    reg = regime()
    rows = []
    for code, g in d.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        ma = pd.Series(c).rolling(50).mean().to_numpy()
        up = (reg.reindex(b.index) == "상승장").to_numpy()
        for _, x in g.iterrows():
            e = int(b.index.searchsorted(x.entry_date))
            t = e - 1
            stop0 = c[t] - atr[t]
            row = {"code": code, "entry_date": x.entry_date, "R_now": x.R2, "f_days": np.nan, "R": np.nan, "win2": np.nan}
            for f in range(t + 1, min(t + WAIT, len(c) - 2) + 1):
                if l[f] <= stop0 or c[f] < ma[f]:
                    break                                         # 전환 전에 죽음
                if up[f]:
                    e2 = f + 1
                    if o[e2] <= stop0:
                        break
                    sp = max(1 - stop0 / o[e2], 0.01)
                    ret, _ = simulate(o, h, l, c, ma, atr, e2, stop0, EXITS["E2 50일선"])
                    row.update(f_days=f - t, R=(ret - vb.COST) / sp, win2=float(ret - vb.COST > 0), buy_date=b.index[e2])
                    break
            rows.append(row)
    ev = pd.DataFrame(rows)
    ev.to_csv(results_dir() / "sideways_flip_events.csv", index=False)
    live = ev[ev.R.notna()].copy()
    res = {"살아 있는 비율": round(len(live) / len(ev), 3), "사건 수": len(ev)}
    for nm, g in {"주 판정: 전환일 지연 매수": live, "㉢ 전환까지 ≤ 5일": live[live.f_days <= 5], "㉢ 6일 이상": live[live.f_days > 5]}.items():
        lv, rb = judge_level(g, rng), robust(g)
        ok = rb.pop("_ok")
        if lv["판정"] != "아님" and not ok:
            lv["판정"] = "아님(②·③ 미달)"
        res[nm] = {**lv, **rb}
    rec = ev[ev.entry_date >= RECENT]
    res["㉠ 사건당 합계 최근(못 산 사건 0R)"] = {"돌파 당일 매수": round(float(rec.R_now.mean()), 3), "전환일 지연 매수": round(float(rec.R.fillna(0).mean()), 3),
                                        "상위 1% 뺀 당일": round(float(rec.R_now.sort_values().iloc[:-max(1, len(rec) // 100)].mean()), 3),
                                        "상위 1% 뺀 지연": round(float(rec.R.fillna(0).sort_values().iloc[:-max(1, len(rec) // 100)].mean()), 3)}
    (results_dir() / "sideways_flip.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
