"""가짜 이탈 — 복귀일 + 2일(3일 봉 모양)을 보고 들어가면 나은가 — 검증이력 9.94 사전 등록 그대로.

  python research/validate_fail_wait3.py   → results/fail_wait3.json
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
import validate_rebreakout as vr
from box_rules import BOX
from reclassify_recent import RECENT, HALF
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

FILTERS = ("F0 기다리기만", "F1 3일째 종가 > 복귀 종가", "F2 양봉 2개 이상", "F3 박스 하단 위 유지", "F4 F1 + F3")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "box_scenario_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    d = d[d.pattern == "failure"].copy()
    d["code"] = d["code"].str.zfill(6)
    rows = []
    for code, g in d.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        Ls = b["low"].rolling(BOX).min().shift(1).to_numpy()
        for _, x in g.iterrows():
            e = int(x.e)
            if e >= len(b) or b.index[e] != x.entry_date:
                continue
            r, bb = e - 1, int(x.b)
            stop, tgt = x.entry * (1 - x.stop_pct), x.entry * (1 + x.tgt_pct)
            row = {"code": code, "entry_date": x.entry_date, "R": (x.ret - vb.COST) / max(x.stop_pct, 0.01), "win": float(x.ret - vb.COST > 0),
                   "stop": float(x.how == "stop")}
            e3 = r + 3
            alive = e3 < len(c) and l[e:e3].min() > stop and h[e:e3].max() < tgt and stop < o[e3] < tgt
            sim = vb.simulate(o, h, l, c, e3, stop, tgt) if alive else None
            if sim is not None:
                ret, _, how = sim
                sp = max(1 - stop / o[e3], 0.01)
                row.update(R3=(ret - vb.COST) / sp, win3=float(ret - vb.COST > 0), stop3=float(how == "stop"))
                f1 = c[r + 2] > c[r]
                f2 = sum(c[k] > o[k] for k in (r, r + 1, r + 2)) >= 2
                f3 = min(l[r + 1], l[r + 2]) > Ls[bb]
                row.update(zip(FILTERS, (True, f1, f2, f3, f1 and f3)))
            rows.append(row)
    ev = pd.DataFrame(rows)
    for f in FILTERS:
        ev[f] = ev[f].fillna(False).astype(bool)
    ev.to_csv(results_dir() / "fail_wait3_events.csv", index=False)
    base = ev[["entry_date", "R", "win", "stop"]]
    per = {"앞": lambda z: z[z.entry_date < RECENT], "최근": lambda z: z[z.entry_date >= RECENT]}
    res = {"기준(복귀 다음날 매수)": {pn: {"n": len(f(base)), "R": round(float(f(base).R.mean()), 3), "승률": round(float(f(base).win.mean()), 3),
                                      "손절률": round(float(f(base).stop.mean()), 3)} for pn, f in per.items()}}
    for fl in FILTERS:
        t = ev[ev[fl]].assign(R=lambda z: z.R3)
        out = {}
        for pn, f in per.items():
            a, bse = f(t), f(base)
            dci = vr.diff_ci(a, bse, rng)
            ev_p = f(ev)
            out[pn] = {"n": len(a), "거래당 R": round(float(a.R.mean()), 3), "차이 [CI]": [round(v, 3) for v in dci],
                       "승률": round(float(a.win3.mean()), 3), "손절률": round(float(a.stop3.mean()), 3),
                       "사건당 R(못 산 0)": round(float(ev_p.R3.where(ev_p[fl], 0).mean()), 3), "사건당 기준": round(float(ev_p.R.mean()), 3)}
        rec = f_rec = t[t.entry_date >= RECENT]
        bse_rec = base[base.entry_date >= RECENT]
        halves = [round(float(rec[rec.entry_date < HALF].R.mean() - bse_rec[bse_rec.entry_date < HALF].R.mean()), 3),
                  round(float(rec[rec.entry_date >= HALF].R.mean() - bse_rec[bse_rec.entry_date >= HALF].R.mean()), 3)]
        o_, r_ = out["앞"], out["최근"]
        same = np.sign(o_["차이 [CI]"][0]) == np.sign(r_["차이 [CI]"][0])
        if fl.startswith("F0"):
            v = "서술"
        elif r_["차이 [CI]"][0] >= 0.10 and r_["차이 [CI]"][1] > 0 and same:
            v = "검증됨" if o_["차이 [CI]"][1] > 0 else ("최근경향" if min(halves) >= 0.05 else "아님(반쪽)")
        else:
            v = "아님"
        if v not in ("아님", "서술", "아님(반쪽)") and r_["사건당 R(못 산 0)"] < r_["사건당 기준"] - 0.05:
            v += " · 거르는 대신 버는 몫이 줄어듦"
        out["반쪽 차이"], out["판정"] = halves, v
        res[fl] = out
    (results_dir() / "fail_wait3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
