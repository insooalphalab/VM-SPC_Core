"""돌파 뒤 N일 유지 확인 후 매수(확인일 종가) — 검증이력 9.75 사전 등록 그대로.

  python research/validate_confirm_entry.py   → results/confirm_entry_validation.json
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
from validate_breakout_exits import EXITS, simulate
from validate_rs_accel import rs_panel
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
COST, E2, HOLD = vb.COST, EXITS["E2 50일선"], 120
DAYS = (5, 10)
LEVELS = ("상단 +10%", "2R", "3R")


def sim_close_entry(h, l, c, o, ma50, d, stop) -> tuple[float, int] | None:
    """d일 종가 매수 → 다음날부터 갭 손절·장중 손절·종가 < 50일선, 최대 HOLD일."""
    X = c[d]
    last = min(d + HOLD, len(c) - 1)
    if d + 1 > last:
        return None
    for j in range(d + 1, last + 1):
        if o[j] <= stop:
            return o[j] / X - 1, j - d
        if l[j] <= stop:
            return stop / X - 1, j - d
        if c[j] < ma50[j]:
            return c[j] / X - 1, j - d
    return c[last] / X - 1, last - d


def run(ev: pd.DataFrame) -> pd.DataFrame:
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
            if t + max(DAYS) + 1 >= len(c):
                continue
            H = h[t - BOX:t].max()
            stop0 = c[t] - atr[t]
            r1 = o[e] - stop0
            ret, hold = simulate(o, h, l, c, ma50, atr, e, stop0, E2)
            row = {"code": code, "entry_date": x.entry_date, "R0": (ret - COST) / max(1 - stop0 / o[e], 0.01)}
            for n in DAYS:
                d = t + n
                alive = hold > d - e + 1                                 # 원 거래가 d일 종가까지 살아 있음
                lv = {"상단 +10%": 1.10 * H, "2R": o[e] + 2 * r1, "3R": o[e] + 3 * r1}
                for name in LEVELS:
                    key = f"{n}일 · {name}"
                    if not (alive and c[d] >= lv[name]):
                        continue
                    stop = c[d] - atr[d]
                    sim = sim_close_entry(h, l, c, o, ma50, d, stop)
                    if sim is None:
                        continue
                    sp = max(1 - stop / c[d], 0.01)
                    row[f"{key}|R"] = (sim[0] - COST) / sp
                    row[f"{key}|win"] = float(sim[0] - COST > 0)
                    row[f"{key}|hold"] = sim[1]
                    row[f"{key}|missed"] = c[d] / o[e] - 1
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    main_ev = pd.read_csv(results_dir() / "breakout_exits_events.csv", parse_dates=["entry_date"], dtype={"code": str})[["code", "entry_date"]]
    main_ev["code"] = main_ev["code"].str.zfill(6)
    allu = pd.read_csv(results_dir() / "entry_trigger_events_all.csv", parse_dates=["entry_date"], dtype={"code": str})
    allu = allu[allu.trig == "A 박스 돌파"][["code", "entry_date"]].copy()
    allu["code"] = allu["code"].str.zfill(6)
    RS = rs_panel()
    pos = RS.index.searchsorted(allu.entry_date) - 1
    allu["rs"] = RS.to_numpy()[pos, RS.columns.get_indexer(allu.code)]
    hold_ev = allu[(allu.rs >= 50) & (allu.rs < 70)][["code", "entry_date"]]
    res = {}
    for scope, ev in (("주 표본 · 상승장 RS ≥ 70", main_ev), ("보지 않은 표본 · 상승장 RS 50~70", hold_ev)):
        d = run(ev)
        d.to_csv(results_dir() / f"confirm_entry_events_{'main' if 'RS ≥ 70' in scope else 'holdout'}.csv", index=False)
        base = d[["entry_date", "R0"]].rename(columns={"R0": "R"})
        r = {"즉시 매수": {"n": int(len(d)), "R": round(float(d.R0.mean()), 3),
                         "맞춤/예측 R": [round(float(base.R[base.entry_date < SPLIT].mean()), 3), round(float(base.R[base.entry_date >= SPLIT].mean()), 3)]}}
        for n in DAYS:
            for name in LEVELS:
                key = f"{n}일 · {name}"
                if f"{key}|R" not in d:
                    continue
                x = d[d[f"{key}|R"].notna()][["entry_date", f"{key}|R", f"{key}|win", f"{key}|hold", f"{key}|missed"]]
                a = x.rename(columns={f"{key}|R": "R"})
                tr = vr.diff_ci(a[a.entry_date < SPLIT], base[base.entry_date < SPLIT], rng)
                te = vr.diff_ci(a[a.entry_date >= SPLIT], base[base.entry_date >= SPLIT], rng)
                r[key] = {"n": int(len(x)), "충족 비율": round(len(x) / len(d), 3), "승률": round(float(x[f"{key}|win"].mean()), 3),
                          "R": round(float(a.R.mean()), 3), "보유일": round(float(x[f"{key}|hold"].mean()), 1),
                          "확인 전 오른 몫(중앙)": round(float(x[f"{key}|missed"].median()), 3),
                          "− 즉시 매수 맞춤": [round(v, 3) for v in tr], "− 즉시 매수 예측": [round(v, 3) for v in te]}
        res[scope] = r
    main_r, hold_r = res["주 표본 · 상승장 RS ≥ 70"], res["보지 않은 표본 · 상승장 RS 50~70"]
    verdict = {}
    for n in DAYS:
        for name in LEVELS:
            key = f"{n}일 · {name}"
            m, hd = main_r.get(key), hold_r.get(key)
            if not m or not hd:
                continue
            te, tr = m["− 즉시 매수 예측"], m["− 즉시 매수 맞춤"]
            ok = te[0] >= 0.10 and te[1] > 0 and tr[0] > 0
            hold_diff = hd["R"] - hold_r["즉시 매수"]["R"]                 # 보지 않은 표본 전체 기간 차이
            verdict[key] = {"보지 않은 표본 차이": round(hold_diff, 3),
                            "판정": "채택" if ok and hold_diff > 0 else "방향만 일치" if (ok or (te[0] > 0 and tr[0] > 0)) else "채택 안 함"}
    res["판정"] = verdict
    (results_dir() / "confirm_entry_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
