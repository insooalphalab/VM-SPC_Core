"""가짜 이탈 복귀일이 강한 반등(≥ +8%)이거나 T² 동반이면 다음날 밀리는가 · 하루 늦게 사면 나은가 — 검증이력 9.95 사전 등록 그대로.

  python research/validate_fail_strong_day.py   → results/fail_strong_day.json
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
from reclassify_recent import level_ci, RECENT, HALF
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

STRONG = 0.08


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "box_scenario_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    d = d[d.pattern == "failure"].copy()
    d["code"] = d["code"].str.zfill(6)
    w = pd.read_csv(results_dir() / "winrate_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    w["code"] = w["code"].str.zfill(6)
    d = d.merge(w[["code", "entry_date", "t2"]].drop_duplicates(["code", "entry_date"]), on=["code", "entry_date"], how="left")
    rows = []
    for code, g in d.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        for _, x in g.iterrows():
            e = int(x.e)
            if e >= len(b) or b.index[e] != x.entry_date or e + 11 >= len(c):
                continue
            r = e - 1
            stop, tgt = x.entry * (1 - x.stop_pct), x.entry * (1 + x.tgt_pct)
            row = {"code": code, "entry_date": x.entry_date, "t2": x.t2 == True, "up_r": c[r] / c[r - 1] - 1,
                   "R": (x.ret - vb.COST) / max(x.stop_pct, 0.01),
                   "d1_red": float(c[e] < o[e]), "d1_below": float(c[e] < c[r]), "d1_oc": c[e] / o[e] - 1,
                   "p5": c[e + 4] / o[e] - 1, "p10": c[e + 9] / o[e] - 1, "Rw": 0.0}
            e2 = e + 1
            if l[e] > stop and h[e] < tgt and stop < o[e2] < tgt:
                sim = vb.simulate(o, h, l, c, e2, stop, tgt)
                if sim is not None:
                    row["Rw"] = (sim[0] - vb.COST) / max(1 - stop / o[e2], 0.01)
            rows.append(row)
    ev = pd.DataFrame(rows)
    ev["S1"] = ev.up_r >= STRONG
    ev["S2"] = ev.t2
    ev["rest"] = ~ev.S1 & ~ev.S2
    ev.to_csv(results_dir() / "fail_strong_day_events.csv", index=False)
    res = {}
    for gname in ("S1", "S2", "rest"):
        g = ev[ev[gname]]
        out = {"n": len(g), "다음날 음봉(시가 매수 후 종가 < 시가)": round(float(g.d1_red.mean()), 3),
               "다음날 종가 < 복귀 종가": round(float(g.d1_below.mean()), 3), "다음날 시가→종가 평균": f"{g.d1_oc.mean():+.2%}",
               "5일째 종가(시가 대비) 평균": f"{g.p5.mean():+.2%}", "10일째": f"{g.p10.mean():+.2%}"}
        if gname != "rest":
            per = {}
            for pn, m in (("앞", g.entry_date < RECENT), ("최근", g.entry_date >= RECENT)):
                y = g[m].assign(R=lambda z: z.Rw - z.R)
                per[pn] = {"n": len(y), "지금 R": round(float(g[m].R.mean()), 3), "하루 늦게 R(못 산 0)": round(float(g[m].Rw.mean()), 3),
                           "차이 [CI]": [round(v, 3) for v in level_ci(y, rng)] if len(y) > 30 else None}
            rec = g[g.entry_date >= RECENT]
            halves = [round(float((rec.Rw - rec.R)[rec.entry_date < HALF].mean()), 3), round(float((rec.Rw - rec.R)[rec.entry_date >= HALF].mean()), 3)]
            o_, r_ = per["앞"]["차이 [CI]"], per["최근"]["차이 [CI]"]
            if r_ and r_[0] >= 0.10 and r_[1] > 0 and np.sign(o_[0]) == np.sign(r_[0]):
                v = "검증됨" if o_[1] > 0 else ("최근경향" if min(halves) >= 0.05 else "아님(반쪽)")
            else:
                v = "아님"
            out.update(하루늦게=per, 반쪽차이=halves, 판정=v)
        res[gname] = out
    (results_dir() / "fail_strong_day.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
