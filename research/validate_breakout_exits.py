"""돌파 후 청산 방식 비교 × 돌파 양봉 크기 — 검증이력 9.72 사전 등록 그대로.

사건 = 9.68 A 박스 돌파(상승장 · RS ≥ 70). 진입·첫 손절 고정, 청산만 바꿔 같은 거래끼리 짝 비교.
  python research/validate_breakout_exits.py   → results/breakout_exits_validation.json
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
from validate_trend_principles import paired_ci
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
COST = vb.COST
# 이름: (청산 이평, 추적 k, 본전, 부분 익절, 시간 손절, 최대 보유)
EXITS = {
    "E0 현행 20일선·60일": (20, None, False, None, False, 60),
    "E0b 20일선·120일": (20, None, False, None, False, 120),
    "E1 10일선": (10, None, False, None, False, 120),
    "E2 50일선": (50, None, False, None, False, 120),
    "E3 추적 2ATR": (None, 2.0, False, None, False, 120),
    "E4 추적 3ATR": (None, 3.0, False, None, False, 120),
    "E5 본전 손절": (20, None, True, None, False, 120),
    "E6 +5% 절반 익절": (20, None, False, ("pct", 0.05), False, 120),
    "E7 +2R 절반 익절": (20, None, False, ("R", 2.0), False, 120),
    "E8 시간 손절 10일": (20, None, False, None, True, 120),
}
BODY_BINS, BODY_LAB = [-np.inf, 0, 0.5, 1.0, np.inf], ["음봉·보합", "작은 양봉(0~0.5ATR)", "중간(0.5~1ATR)", "장대(≥1ATR)"]


def simulate(o, h, l, c, ma, atr, e, stop0, cfg) -> tuple[float, int]:
    """(블렌딩 수익, 보유일). 하루 안 순서: 갭 손절 → 장중 손절 → 부분 익절 → 종가 청산 → (장 마감 뒤) 고점·추적·본전 갱신."""
    ma_n, trail, be, part, tstop, hold = cfg
    entry, s, peak = o[e], stop0, o[e]
    r1 = entry - stop0
    tgt = None if part is None else entry * (1 + part[1]) if part[0] == "pct" else entry + part[1] * r1
    taken, part_ret = False, 0.0
    last = min(e + hold, len(c)) - 1

    def done(px, j):
        rest = px / entry - 1
        return (0.5 * part_ret + 0.5 * rest if taken else rest), j - e + 1

    for j in range(e, last + 1):
        if j > e and o[j] <= s:
            return done(o[j], j)
        if l[j] <= s:
            return done(s, j)
        if tgt is not None and not taken and h[j] >= tgt:
            taken, part_ret = True, max(tgt, o[j]) / entry - 1
        if ma is not None and c[j] < ma[j]:
            return done(c[j], j)
        if tstop and j == e + 9 and max(peak, h[j]) < entry + r1:
            return done(c[j], j)
        peak = max(peak, h[j])
        if trail is not None and not np.isnan(atr[j]):
            s = max(s, peak - trail * atr[j])
        if be and peak >= entry + r1:
            s = max(s, entry)
    return done(c[last], last)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "entry_trigger_events_rs.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev = ev[ev.trig == "A 박스 돌파"].copy()
    ev["code"] = ev["code"].str.zfill(6)
    rows = []
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        mas = {n: pd.Series(c).rolling(n).mean().to_numpy() for n in (10, 20, 50)}
        for _, x in g.iterrows():
            e = int(b.index.searchsorted(x.entry_date))
            t = e - 1
            stop0 = c[t] - atr[t]
            sp = max(1 - stop0 / o[e], 0.01)
            row = {"code": code, "entry_date": x.entry_date, "body": (c[t] - o[t]) / atr[t]}
            for name, cfg in EXITS.items():
                ret, hold = simulate(o, h, l, c, mas.get(cfg[0]), atr, e, stop0, cfg)
                row[f"{name}|R"] = (ret - COST) / sp
                row[f"{name}|win"] = float(ret - COST > 0)
                row[f"{name}|hold"] = hold
            rows.append(row)
    d = pd.DataFrame(rows)
    d["body_b"] = pd.cut(d.body, BODY_BINS, labels=BODY_LAB)
    d.to_csv(results_dir() / "breakout_exits_events.csv", index=False)
    fit, test = d.entry_date < SPLIT, d.entry_date >= SPLIT

    def table(x):
        out = {}
        for name in EXITS:
            r = x[f"{name}|R"]
            top = r.sort_values(ascending=False).head(max(1, len(r) // 10)).sum()
            out[name] = {"승률": round(float(x[f"{name}|win"].mean()), 3), "R": round(float(r.mean()), 3),
                         "맞춤 R": round(float(r[fit[x.index]].mean()), 3), "예측 R": round(float(r[test[x.index]].mean()), 3),
                         "보유일": round(float(x[f"{name}|hold"].mean()), 1),
                         "상위10% 몫": round(float(top / r.sum()), 2) if r.sum() > 0 else None}
        return out

    def judge(x):
        base = "E0 현행 20일선·60일"
        cand = {n: float((x.loc[fit[x.index], f"{n}|R"] - x.loc[fit[x.index], f"{base}|R"]).mean()) for n in EXITS if n != base}
        pick = max(cand, key=cand.get)
        y = x.assign(dR=x[f"{pick}|R"] - x[f"{base}|R"])
        tr, te = paired_ci(y[fit[x.index]], rng), paired_ci(y[test[x.index]], rng)
        return {"고른 청산": pick, "맞춤 차이": [round(v, 3) for v in tr], "예측 차이": [round(v, 3) for v in te],
                "verdict": "교체" if te[0] >= 0.10 and te[1] > 0 and tr[0] > 0 else "방향만 일치" if te[0] > 0 and tr[0] > 0 else "현행 유지"}

    res = {"n": int(len(d)), "전체": {"표": table(d), "판정": judge(d)}, "양봉 크기별": {}}
    for bname, x in d.groupby("body_b"):
        res["양봉 크기별"][str(bname)] = {"n": int(len(x)), "표": table(x), "판정": judge(x)}
    (results_dir() / "breakout_exits_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"n={len(d)}")
    for scope, x in [("전체", d)] + [(str(k), v) for k, v in d.groupby("body_b")]:
        t = table(x)
        print(f"\n== {scope} (n={len(x)})")
        print(pd.DataFrame(t).T.to_string())
        print("판정:", judge(x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
