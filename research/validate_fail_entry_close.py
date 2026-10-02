"""가짜 이탈 — 복귀 확인 당일 종가(장외) 매수 vs 다음날 시가 매수 — 검증이력 9.107 사전 등록 그대로.

  python research/validate_fail_entry_close.py   → results/fail_entry_close.json
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

from build_winrate_table import pf
from reclassify_recent import level_ci
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

COST, HOLD = 0.003, 20
SPLIT, HALF, END = pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
REGS = ("횡보·전환", "하락장", "상승장")


def run(o, h, l, c, entry: float, first: int, stop: float, tgt: float) -> float:
    """first일부터 HOLD일. 시가 갭 먼저, 같은 날 둘 다면 손절."""
    for j in range(first, first + HOLD):
        if o[j] <= stop:
            return o[j] / entry - 1
        if o[j] >= tgt:
            return o[j] / entry - 1
        if l[j] <= stop:
            return stop / entry - 1
        if h[j] >= tgt:
            return tgt / entry - 1
    return c[first + HOLD - 1] / entry - 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    w = pd.read_csv(results_dir() / "winrate_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    w["code"] = w.code.str.zfill(6)
    rows = []
    for code, g in w.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        if b is None:
            continue
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        for _, x in g.iterrows():
            e = int(b.index.searchsorted(x.entry_date))                   # r + 1
            if e < 1 or e + HOLD > len(c) or b.index[e] != x.entry_date:
                continue
            r = e - 1
            stop, tgt = o[e] * (1 - x.stop_pct), o[e] * (1 + x.tgt_pct)   # 사건표의 손절 · 목표 가격 복원
            if not stop < c[r] < tgt:
                continue
            ra = run(o, h, l, c, c[r], e, stop, tgt)
            Ra = (ra - COST) / max(1 - stop / c[r], 0.01)
            if stop < o[e] < tgt:
                rb = run(o, h, l, c, o[e], e, stop, tgt)
                Rb = (rb - COST) / max(1 - stop / o[e], 0.01)
            else:
                Rb = 0.0
            rows.append({"code": code, "entry_date": x.entry_date, "reg": x.reg, "Ra": Ra, "Rb": Rb, "gap": o[e] / c[r] - 1})
    d = pd.DataFrame(rows).assign(R=lambda z: z.Ra - z.Rb)
    d.to_csv(results_dir() / "fail_entry_close_events.csv", index=False)
    res = {}
    for rg in REGS:
        x = d[d.reg == rg]
        per = lambda lo, hi: x[(x.entry_date >= lo) & (x.entry_date < hi)]
        o_, r_ = per(pd.Timestamp("2000-01-01"), SPLIT), per(SPLIT, END)
        oc, rc = level_ci(o_, rng), level_ci(r_, rng)
        h1, h2 = float(per(SPLIT, HALF).R.mean()), float(per(HALF, END).R.mean())
        excl = lambda q: q[1] > 0 or q[2] < 0
        if abs(rc[0]) >= 0.05 and excl(rc) and abs(oc[0]) >= 0.05 and excl(oc) and np.sign(oc[0]) == np.sign(rc[0]):
            v = "검증됨"
        elif abs(rc[0]) >= 0.05 and excl(rc) and np.sign(h1) == np.sign(h2) == np.sign(rc[0]):
            v = "최근경향"
        else:
            v = "아님"
        res[rg] = {"건수": len(x), "A − B 앞 [CI]": [round(q, 3) for q in oc], "A − B 최근 [CI]": [round(q, 3) for q in rc],
                   "반쪽": [round(h1, 3), round(h2, 3)],
                   "밤사이 갭 평균 · 위로 뜬 비율": [f"{x.gap.mean():+.2%}", round(float((x.gap > 0).mean()), 3)],
                   "A 종가 매수 승률 · R · 수익 배수": [round(float((x.Ra > 0).mean()), 3), round(float(x.Ra.mean()), 3), pf(x.Ra)],
                   "B 시가 매수 승률 · R · 수익 배수": [round(float((x.Rb > 0).mean()), 3), round(float(x.Rb.mean()), 3), pf(x.Rb)],
                   "판정": v}
    (results_dir() / "fail_entry_close.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
