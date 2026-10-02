"""업종 대비 과하게 빠진 종목은 업종 평균으로 돌아오는가 — 검증이력 9.104 사전 등록 그대로.

  python research/validate_industry_reversion.py   → results/industry_reversion.json
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
from validate_industry_action import sectors
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

START, SPLIT, HALF, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")
MIN_SECTOR, GAP, ZCUT = 5, 10, 2.0
REGS = ("상승장", "횡보·전환", "하락장")


def peer_mean(X: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """같은 업종 다른 종목 평균(자기 제외)."""
    out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for _, cols in groups.groupby(groups).groups.items():
        sub = X[list(cols)]
        s, n = sub.sum(axis=1, min_count=1), sub.notna().sum(axis=1)
        out[list(cols)] = (s.to_numpy()[:, None] - sub.fillna(0).to_numpy()) / (n.to_numpy()[:, None] - sub.notna().to_numpy())
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    sec = sectors()
    O, C = {}, {}
    for code in validation_codes():
        if code not in sec:
            continue
        b = load_bars(LONG_HISTORY, code)
        if b is None or len(b) < 300:
            continue
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        O[code], C[code] = b["open"], b["close"]
    C, O = pd.DataFrame(C).sort_index(), pd.DataFrame(O).sort_index()
    groups = pd.Series({c: sec[c] for c in C.columns})
    size = groups.map(groups.value_counts())
    keep = size[size >= MIN_SECTOR].index
    C, O, groups = C[keep], O[keep], groups[keep]
    r5 = C / C.shift(5) - 1
    res5 = r5 - peer_mean(r5, groups)
    z = res5 / res5.rolling(250, min_periods=200).std().shift(1)
    fw = {}
    for k in (5, 10, 20):
        f = C.shift(-k) / O.shift(-1) - 1                                   # t+1 시가 → t+k 종가
        fw[k] = (f - peer_mean(f, groups), f)
    idx = load_bars(LONG_HISTORY, "069500")
    ifw = {k: (idx["close"].shift(-k) / idx["open"].shift(-1) - 1).reindex(C.index) for k in (5, 10, 20)}
    reg = regime().reindex(C.index)
    rows = []
    Z = z.to_numpy()
    for j, code in enumerate(C.columns):
        last = -GAP
        for side, cond in (("down", Z[:, j] <= -ZCUT), ("up", Z[:, j] >= ZCUT)):
            last = -GAP
            for t in np.flatnonzero(cond):
                d = C.index[t]
                if d < START or t - last < GAP or t + 21 >= len(C):
                    continue
                last = t
                row = {"code": code, "entry_date": C.index[t + 1], "reg": reg.iloc[t], "side": side, "z": Z[t, j]}
                for k in (5, 10, 20):
                    row[f"rel{k}"] = fw[k][0].iat[t, j] * 100
                    row[f"idx{k}"] = (fw[k][1].iat[t, j] - ifw[k].iat[t]) * 100
                rows.append(row)
    d = pd.DataFrame(rows).dropna(subset=["rel10"])
    d.to_csv(results_dir() / "industry_reversion_events.csv", index=False)

    def level(x, col):
        x = x.assign(R=x[col])
        per = lambda lo, hi: x[(x.entry_date >= lo) & (x.entry_date < hi)]
        o, r = per(START, SPLIT), per(SPLIT, END)
        oc = level_ci(o, rng) if len(o) >= 30 else (np.nan,) * 3
        rc = level_ci(r, rng) if len(r) >= 30 else (np.nan,) * 3
        h1, h2 = float(per(SPLIT, HALF).R.mean()), float(per(HALF, END).R.mean())
        if rc[0] >= 1.0 and rc[1] > 0 and min(h1, h2) >= 0.5:
            v = "검증됨" if oc[0] >= 1.0 and oc[1] > 0 else "최근경향"
        elif oc[0] > 0 and rc[0] > 0 and h1 > 0 and h2 > 0:
            v = "약함(네 구간 같은 방향)"
        elif oc[0] < 0 and rc[0] < 0 and h1 < 0 and h2 < 0:
            v = "반대: 이어짐(네 구간 모두 마이너스)"
        else:
            v = "아님"
        return {"건수 앞/최근": [len(o), len(r)], "앞 [CI]": [round(q, 2) for q in oc], "최근 [CI]": [round(q, 2) for q in rc],
                "반쪽": [round(h1, 2), round(h2, 2)], "플러스 비율": round(float((x.R > 0).mean()), 3), "판정": v}

    res = {"업종 수": int(groups.nunique()), "종목 수": len(groups)}
    for rg in REGS:
        x = d[(d.reg == rg) & (d.side == "down")]
        u = d[(d.reg == rg) & (d.side == "up")]
        res[rg] = {"주: 과하게 빠짐 → 10일 업종 대비 %p": level(x, "rel10"),
                   "서술: 5 · 20일 업종 대비": [round(float(x.rel5.mean()), 2), round(float(x.rel20.mean()), 2)],
                   "서술: 10일 지수 대비": round(float(x.idx10.mean()), 2),
                   "서술: 과하게 오름 → 10일 업종 대비": level(u, "rel10")}
    (results_dir() / "industry_reversion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
