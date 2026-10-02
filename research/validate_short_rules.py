"""짧은 호흡(10거래일 안) 규칙 — 기대값 +0.2R 이상인 칸이 있는가 — 검증이력 9.102 사전 등록 그대로.

  python research/validate_short_rules.py   → results/short_rules.json
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

from reclassify_recent import level_ci
from universe_all import validation_codes
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

COST, HOLD = 0.003, 10
START, SPLIT, HALF, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
REGS = ("상승장", "횡보·전환", "하락장")


def sim(o, h, l, c, e: int, stop: float, tgt: float):
    """e일 시가 진입, 최대 HOLD일. (수익, 보유일, 결과). 같은 날 둘 다 닿으면 손절."""
    entry = o[e]
    for j in range(e, e + HOLD):
        if j > e and o[j] <= stop:
            return o[j] / entry - 1, j - e + 1, "손절"
        if j > e and o[j] >= tgt:
            return o[j] / entry - 1, j - e + 1, "목표"
        if l[j] <= stop:
            return stop / entry - 1, j - e + 1, "손절"
        if h[j] >= tgt:
            return tgt / entry - 1, j - e + 1, "목표"
    return c[e + HOLD - 1] / entry - 1, HOLD, "만기"


def bars_of(code: str):
    b = load_bars(LONG_HISTORY, code)
    if b is None or len(b) < 300:
        return None
    return b[(b[["open", "high", "low", "close", "volume"]] > 0).all(1)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    rows = []
    # K1~K4 가짜 이탈
    w = pd.read_csv(results_dir() / "winrate_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    w["code"] = w.code.str.zfill(6)
    for code, g in w.groupby("code"):
        b = bars_of(code)
        if b is None:
            continue
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        for _, x in g.iterrows():
            e = int(b.index.searchsorted(x.entry_date))
            if e >= len(b) - HOLD or b.index[e] != x.entry_date or e < 2:
                continue
            entry = o[e]
            stop, tgt = entry * (1 - x.stop_pct), entry * (1 + x.tgt_pct)
            if not stop < entry < tgt:
                continue
            ret, days, how = sim(o, h, l, c, e, stop, tgt)
            base = {"code": code, "entry_date": x.entry_date, "reg": x.reg, "R": (ret - COST) / max(x.stop_pct, 0.01),
                    "ret": ret, "days": days, "how": how}
            rows.append({**base, "rule": "K1 가짜 이탈 전체"})
            if str(x.t2) == "True":
                rows.append({**base, "rule": "K2 + T² 동반"})
            if x.depth >= 0.035:
                rows.append({**base, "rule": "K3 + 깊이 ≥ 3.5%"})
            if c[e - 1] / c[e - 2] - 1 >= 0.08:
                rows.append({**base, "rule": "K4 + 복귀일 ≥ +8%"})
    # K5 거래량 가뭄 양봉
    for code in validation_codes():
        b = bars_of(code)
        if b is None:
            continue
        o, h, l, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        va = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        first = max(int(b.index.searchsorted(START)), 21)
        last = -HOLD
        for t in range(first, len(c) - HOLD - 1):
            if t - last < HOLD:                                   # 같은 종목 거래 겹침 방지
                continue
            if not (c[t] / c[t - 1] - 1 >= 0.03 and c[t] > o[t] and va[t] > 0 and v[t] <= 0.5 * va[t]):
                continue
            e = t + 1
            stop = l[t]
            if not stop < o[e]:
                continue
            sp = 1 - stop / o[e]
            tgt = o[e] * (1 + 3 * sp)
            ret, days, how = sim(o, h, l, c, e, stop, tgt)
            last = t
            rows.append({"code": code, "entry_date": b.index[e], "reg": reg.get(b.index[t], np.nan), "R": (ret - COST) / max(sp, 0.01),
                         "ret": ret, "days": days, "how": how, "rule": "K5 거래량 가뭄 양봉"})
    d = pd.DataFrame(rows)
    d.to_csv(results_dir() / "short_rules_events.csv", index=False)
    res = {}
    for (rule, rg), x in d.groupby(["rule", "reg"]):
        if rg not in REGS:
            continue
        o_, r_ = x[x.entry_date < SPLIT], x[x.entry_date >= SPLIT]
        oc = level_ci(o_, rng) if len(o_) >= 30 else (np.nan,) * 3
        rc = level_ci(r_, rng) if len(r_) >= 30 else (np.nan,) * 3
        h1, h2 = float(r_[r_.entry_date < HALF].R.mean()), float(r_[r_.entry_date >= HALF].R.mean())
        pick = oc[0] >= 0.20 and oc[1] > 0
        conf = rc[0] >= 0.20 and rc[1] > 0
        v = "검증됨" if pick and conf else ("최근경향" if conf and min(h1, h2) >= 0.10 else ("앞에서만" if pick else "아님"))
        win, loss = x[x.R > 0].R, x[x.R <= 0].R
        res[f"{rule} | {rg}"] = {
            "건수 앞/최근": [len(o_), len(r_)], "앞 R [CI]": [round(q, 3) for q in oc], "최근 R [CI]": [round(q, 3) for q in rc],
            "반쪽": [round(h1, 3), round(h2, 3)], "승률": round(float((x.R > 0).mean()), 3),
            "목표 · 손절 · 만기": [round(float((x.how == k).mean()), 3) for k in ("목표", "손절", "만기")],
            "평균 보유일": round(float(x.days.mean()), 1), "평균 이익 / 손실 R": [round(float(win.mean()), 2), round(float(loss.mean()), 2)], "판정": v}
    (results_dir() / "short_rules.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
