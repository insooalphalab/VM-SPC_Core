"""돌파 기대값(R) 모형 — 검증이력 9.54 사전 등록 기준 그대로.

  python research/validate_breakout_r.py   → results/breakout_r_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

from v2_config import results_dir

SPLIT = pd.Timestamp("2021-06-01")
CLIP = (-2.0, 10.0)
BASE = ["reg_side", "reg_down"]
CANDS = {"종목 변동성": ["vol250"], "돌파 강도": ["strength"], "변동성 × 상승장": ["vol250", "vol_bull"]}


def fit_predict(tr, te, cols):
    Xtr = np.c_[np.ones(len(tr)), tr[cols].to_numpy(float)]
    Xte = np.c_[np.ones(len(te)), te[cols].to_numpy(float)]
    b, *_ = np.linalg.lstsq(Xtr, tr["Rc"].to_numpy(float), rcond=None)
    return Xte @ b, b


def spread(te, pred, rng) -> tuple[float, float, float]:
    """예측 상위 1/3 − 하위 1/3 의 실제 R 차이(같은 날 예측이 같을 수 있어 순위 기준), 20일 블록 부트스트랩."""
    t = te.assign(p=pred)
    if t["p"].nunique() <= 3:                         # 장세만 쓴 기본 모형은 값이 3개뿐 — 가장 높은 장세 vs 가장 낮은 장세로 비교
        t = t.assign(g=np.where(t.p == t.p.max(), 1, np.where(t.p == t.p.min(), 0, -1)))
    else:
        q1, q2 = t["p"].quantile([1 / 3, 2 / 3])
        t = t.assign(g=np.where(t.p > q2, 1, np.where(t.p <= q1, 0, -1)))
    t = t[t.g >= 0]
    t["block"] = t.entry_date.rank(method="dense").astype(int) // 20
    agg = np.array([(x.loc[x.g == 1, "R"].sum(), (x.g == 1).sum(), x.loc[x.g == 0, "R"].sum(), (x.g == 0).sum())
                    for _, x in t.groupby("block")], float)
    idx = rng.integers(0, len(agg), (2000, len(agg)))
    s = agg[idx].sum(1)
    with np.errstate(all="ignore"):
        bs = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(t[t.g == 1].R.mean() - t[t.g == 0].R.mean()), float(lo), float(hi)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "breakout_edge_events.csv", parse_dates=["entry_date"]).dropna(subset=["vol250", "strength", "reg"])
    d["Rc"] = d["R"].clip(*CLIP)
    d["reg_side"] = (d.reg == "횡보·전환").astype(float)
    d["reg_down"] = (d.reg == "하락장").astype(float)
    d["vol_bull"] = d.vol250 * (d.reg == "상승장")
    tr, te = d[d.entry_date < SPLIT], d[d.entry_date >= SPLIT]
    res = {"n_train": int(len(tr)), "n_test": int(len(te))}
    pb, _ = fit_predict(tr, te, BASE)
    base = spread(te, pb, rng)
    res["기본(장세만)"] = {"상위-하위 R": round(base[0], 3), "ci": [round(base[1], 3), round(base[2], 3)]}
    for name, cols in CANDS.items():
        p, b = fit_predict(tr, te, BASE + cols)
        sp = spread(te, p, rng)
        res[name] = {"상위-하위 R": round(sp[0], 3), "ci": [round(sp[1], 3), round(sp[2], 3)], "coef": [round(float(x), 3) for x in b],
                     "verdict": "남김" if sp[0] - base[0] >= 0.10 and sp[1] > 0 else "뺌"}
    # 서술: 장세별, 상승장 연도별 변동성 상위-하위 R
    q1, q2 = tr.vol250.quantile([1 / 3, 2 / 3])
    hl = lambda g: (float(g[g.vol250 > q2].R.mean() - g[g.vol250 <= q1].R.mean()), int((g.vol250 > q2).sum()), int((g.vol250 <= q1).sum()))
    res["장세별 변동성 상위-하위 R"] = {r: [round(x, 3) if isinstance(x, float) else x for x in hl(g)] for r, g in d.groupby("reg")}
    bull = d[d.reg == "상승장"]
    res["상승장 연도별 변동성 상위-하위 R"] = {int(y): [round(x, 3) if isinstance(x, float) else x for x in hl(g)]
                                    for y, g in bull.groupby(bull.entry_date.dt.year) if len(g) > 300}
    (results_dir() / "breakout_r_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
