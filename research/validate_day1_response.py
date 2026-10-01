"""돌파 다음날 움직임(돌파 폭 단위) → 들고 갈지 끊을지 — 검증이력 9.73 사전 등록 그대로.

  python research/validate_day1_response.py   → results/day1_response_validation.json
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
from box_rules import BOX
from validate_breakout_exits import EXITS, simulate
from validate_trend_principles import paired_ci
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
COST, E2 = vb.COST, EXITS["E2 50일선"]
Z_BINS, Z_LAB = [-np.inf, -1, 0, 1, np.inf], ["≤ −1 (박스 안 복귀)", "−1 ~ 0 (밀렸지만 상단 위)", "0 ~ 1", "> 1 (강한 연장)"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "breakout_exits_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    rows = []
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        ma50 = pd.Series(c).rolling(50).mean().to_numpy()
        for _, x in g.iterrows():
            e = int(b.index.searchsorted(x.entry_date))
            t = e - 1
            if e + 1 >= len(c):
                continue
            H = h[t - BOX:t].max()
            stop0 = c[t] - atr[t]
            sp = max(1 - stop0 / o[e], 0.01)
            ret, hold = simulate(o, h, l, c, ma50, atr, e, stop0, E2)
            R2 = (ret - COST) / sp
            row = {"code": code, "entry_date": x.entry_date, "R_E2": R2, "win_E2": float(ret - COST > 0)}
            if hold == 1:                                              # 다음날(진입일) 손절·50일선 청산
                row.update(bucket="다음날 손절·청산", z=np.nan, z_atr=np.nan, R_X=R2, rem=np.nan)
            else:
                z = (c[e] - c[t]) / max(c[t] - H, 0.1 * atr[t])
                ret_t2 = o[e + 1] / o[e] - 1
                rx = (ret_t2 - COST) / sp if z <= -1 else R2
                row.update(bucket=None, z=z, z_atr=(c[e] - c[t]) / atr[t], R_X=rx, rem=(ret - ret_t2) / sp)
            rows.append(row)
    d = pd.DataFrame(rows)
    d.loc[d.bucket.isna(), "bucket"] = pd.cut(d.z, Z_BINS, labels=Z_LAB).astype(str)
    d["bucket_atr"] = pd.cut(d.z_atr, Z_BINS, labels=["≤ −1ATR", "−1 ~ 0ATR", "0 ~ 1ATR", "> 1ATR"]).astype(str)
    d.to_csv(results_dir() / "day1_response_events.csv", index=False)
    fit, test = d.entry_date < SPLIT, d.entry_date >= SPLIT

    def tab(col):
        out = {}
        for k, x in d.groupby(col):
            out[k] = {"n": int(len(x)), "비중": round(len(x) / len(d), 3), "승률(E2)": round(float(x.win_E2.mean()), 3),
                      "R(E2)": round(float(x.R_E2.mean()), 3), "맞춤 R": round(float(x.R_E2[fit].mean()), 3), "예측 R": round(float(x.R_E2[test].mean()), 3),
                      "+2일 시가부터 남은 R": round(float(x.rem.mean()), 3) if x.rem.notna().any() else None,
                      "남은 R 맞춤/예측": [round(float(x.rem[fit].mean()), 3), round(float(x.rem[test].mean()), 3)] if x.rem.notna().any() else None}
        return out

    y = d.assign(dR=d.R_X - d.R_E2)
    tr, te = paired_ci(y[fit], rng), paired_ci(y[test], rng)
    res = {"n": int(len(d)), "돌파 폭 단위": tab("bucket"), "ATR 단위": tab("bucket_atr"),
           "X 복귀 컷 − E2": {"맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                            "verdict": "채택" if te[0] >= 0.05 and te[1] > 0 and tr[0] > 0 else
                                       "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}}
    (results_dir() / "day1_response_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
