"""RS 가속도(최근 1개월 RS 순위 급상승) 돌파 — 검증이력 9.64 사전 등록 그대로.

  python research/validate_rs_accel.py   → results/rs_accel_validation.json
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
from universe_all import validation_codes
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
RS_CUT, ACCEL, LAG = 70, 15, 20
STOP_R = -0.95


def rs_panel() -> pd.DataFrame:
    """9.62와 같은 RS: 0.4·r63 + 0.2·r126 + 0.2·r189 + 0.2·r252 의 791종목 안 백분위(0~100)."""
    rs = {}
    for code in validation_codes():
        b = load_bars(LONG_HISTORY, code)
        if b is None or len(b) < 300:
            continue
        c = b.loc[(b[["open", "high", "low", "close"]] > 0).all(1), "close"]
        rs[code] = 0.4 * (c / c.shift(63) - 1) + 0.2 * (c / c.shift(126) - 1) + 0.2 * (c / c.shift(189) - 1) + 0.2 * (c / c.shift(252) - 1)
    return pd.DataFrame(rs).sort_index().rank(axis=1, pct=True) * 100


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    RS = rs_panel()
    ev = pd.read_csv(results_dir() / "trend_principles_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    ev = ev[ev.code.isin(RS.columns)].copy()
    pos = RS.index.searchsorted(ev.entry_date) - 1                       # 돌파일(시장 공통 거래일)
    ok = pos >= LAG
    ev, pos = ev[ok].copy(), pos[ok]
    col = RS.columns.get_indexer(ev.code)
    arr = RS.to_numpy()
    ev["rs_now"] = arr[pos, col]
    ev["d_rs"] = ev.rs_now - arr[pos - LAG, col]
    ev = ev.dropna(subset=["rs_now", "d_rs"])
    ev["stopped"] = ev.R <= STOP_R
    ev.to_csv(results_dir() / "rs_accel_events.csv", index=False)

    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "손절 종료": round(float(g.stopped.mean()), 3)}
    res = {"장세별": {}}
    for r, g in ev.groupby("reg"):
        hi, lo = g[g.rs_now >= RS_CUT], g[g.rs_now < RS_CUT]
        res["장세별"][r] = {"RS≥70 ΔRS<0(감속)": s(hi[hi.d_rs < 0]), "RS≥70 ΔRS 0~15": s(hi[(hi.d_rs >= 0) & (hi.d_rs < ACCEL)]),
                           "RS≥70 ΔRS≥15(가속)": s(hi[hi.d_rs >= ACCEL]), "RS<70 ΔRS≥15": s(lo[lo.d_rs >= ACCEL]),
                           "RS<70 나머지": s(lo[lo.d_rs < ACCEL])}
    u = ev[(ev.reg == "상승장") & (ev.rs_now >= RS_CUT)]
    for name, flag, mode in (("가속(ΔRS≥15)", u.d_rs >= ACCEL, "up"), ("감속(ΔRS<0)", u.d_rs < 0, "down")):
        a, b = u[flag], u[~flag]
        tr = vr.diff_ci(a[a.entry_date < SPLIT], b[b.entry_date < SPLIT], rng)
        te = vr.diff_ci(a[a.entry_date >= SPLIT], b[b.entry_date >= SPLIT], rng)
        if mode == "up":
            v = "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"
        else:
            v = "감속 경고 채택" if te[2] < 0 and tr[0] < 0 else "방향만 일치" if te[0] < 0 and tr[0] < 0 else "채택 안 함"
        res[f"상승장 RS≥70 {name} − 나머지 R"] = {"n": int(flag.sum()), "맞춤": [round(x, 3) for x in tr],
                                             "예측": [round(x, 3) for x in te], "verdict": v}
    (results_dir() / "rs_accel_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
