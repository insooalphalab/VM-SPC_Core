"""보유 중 "업종 대비 과열" 경고 — 가짜 이탈 거래가 목표 전에 업종보다 과하게 오르면 그날 파는 게 나은가 — 검증이력 9.105 사전 등록 그대로.

  python research/validate_industry_overheat_exit.py   → results/industry_overheat_exit.json
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
from reclassify_recent import level_ci
from universe_all import validation_codes
from validate_industry_action import sectors
from validate_industry_reversion import MIN_SECTOR, ZCUT, peer_mean
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT, HALF, END = pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
HOLD = 20
REGS = ("상승장", "횡보·전환", "하락장")


def industry_z() -> tuple[pd.DataFrame, dict]:
    sec = sectors()
    bars, C = {}, {}
    for code in validation_codes():
        if code not in sec:
            continue
        b = load_bars(LONG_HISTORY, code)
        if b is None or len(b) < 300:
            continue
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        bars[code], C[code] = b, b["close"]
    C = pd.DataFrame(C).sort_index()
    groups = pd.Series({c: sec[c] for c in C.columns})
    keep = groups[groups.map(groups.value_counts()) >= MIN_SECTOR].index
    C, groups = C[keep], groups[keep]
    r5 = C / C.shift(5) - 1
    res5 = r5 - peer_mean(r5, groups)
    return res5 / res5.rolling(250, min_periods=200).std().shift(1), bars


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    Z, bars = industry_z()
    w = pd.read_csv(results_dir() / "winrate_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    w["code"] = w.code.str.zfill(6)
    rows = []
    for code, g in w.groupby("code"):
        if code not in Z.columns:
            continue
        b = bars[code]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        z = Z[code].reindex(b.index).to_numpy()
        for _, x in g.iterrows():
            e = int(b.index.searchsorted(x.entry_date))
            if e + HOLD > len(c) or b.index[e] != x.entry_date:
                continue
            entry = o[e]
            stop, tgt = entry * (1 - x.stop_pct), entry * (1 + x.tgt_pct)
            if not stop < entry < tgt:
                continue
            warn, final = None, None
            for j in range(e, e + HOLD):
                if j > e and o[j] <= stop:
                    final = o[j] / entry - 1; break
                if j > e and o[j] >= tgt:
                    final = o[j] / entry - 1; break
                if l[j] <= stop:
                    final = stop / entry - 1; break
                if h[j] >= tgt:
                    final = tgt / entry - 1; break
                if warn is None and np.isfinite(z[j]) and z[j] >= ZCUT and j < e + HOLD - 1:
                    warn = (j - e + 1, c[j] / entry - 1)
            if final is None:
                final = c[e + HOLD - 1] / entry - 1
            sp = max(x.stop_pct, 0.01)
            row = {"code": code, "entry_date": x.entry_date, "reg": x.reg, "R": (final - vb.COST) / sp, "warned": warn is not None}
            if warn:
                row.update(warn_day=warn[0], R_at_warn=(warn[1] - vb.COST) / sp, rem=(final - warn[1]) / sp)
            rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(results_dir() / "industry_overheat_exit_events.csv", index=False)
    res = {}
    for rg in REGS:
        x = d[d.reg == rg]
        y = x[x.warned].assign(R=lambda q: q.rem)
        per = lambda lo, hi: y[(y.entry_date >= lo) & (y.entry_date < hi)]
        o_, r_ = per(pd.Timestamp("2000-01-01"), SPLIT), per(SPLIT, END)
        oc = level_ci(o_, rng) if len(o_) >= 30 else (np.nan,) * 3
        rc = level_ci(r_, rng) if len(r_) >= 30 else (np.nan,) * 3
        h1, h2 = float(per(SPLIT, HALF).R.mean()), float(per(HALF, END).R.mean())
        if rc[0] <= -0.15 and rc[2] < 0 and h1 < 0 and h2 < 0:
            v = "검증됨" if oc[2] < 0 else "최근경향"
        elif oc[0] < 0 and rc[0] < 0 and h1 < 0 and h2 < 0:
            v = "약함(네 구간 음수)"
        elif rc[0] >= 0:
            v = "경고 무용(들고 가는 게 맞음)"
        else:
            v = "아님"
        res[rg] = {"거래 수": len(x), "경고 뜬 비율": round(float(x.warned.mean()), 3), "경고일(중앙, 매수 후 n일째)": float(y.warn_day.median()) if len(y) else None,
                   "경고일까지 R(평균)": round(float(y.R_at_warn.mean()), 3) if len(y) else None,
                   "남은 R 앞 [CI]": [round(q, 3) for q in oc], "남은 R 최근 [CI]": [round(q, 3) for q in rc], "반쪽": [round(h1, 3), round(h2, 3)],
                   "경고 없는 거래 R": round(float(x[~x.warned].R.mean()), 3), "판정": v}
    (results_dir() / "industry_overheat_exit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
