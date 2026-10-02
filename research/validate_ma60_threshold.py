"""횡보장 60일선 위치 — 상위 몇 %부터 되돌림이 시작되는가 — 검증이력 9.93 사전 등록 그대로.

  python research/validate_ma60_threshold.py   → results/ma60_threshold.json  (9.80 패널)
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

from reclassify_recent import RECENT, HALF
from validate_compression_regime import regime
from validate_holding_factors import boot
from v2_config import results_dir

EDGES = [0, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]
LAB = ["하위 20", "20~50", "50~70", "70~80", "상위 10~20", "상위 5~10", "상위 5"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "holding_factors_panel.csv", parse_dates=["date"])
    d = d[d.ma60.notna() & d.x20.notna()]
    d["pct"] = d.groupby("date").ma60.rank(pct=True)
    d["bin"] = pd.cut(d.pct, EDGES, labels=LAB, right=True)
    d["reg"] = regime().reindex(d.date).to_numpy()
    res = {}
    for rg in ("횡보·전환", "상승장", "하락장"):
        x = d[d.reg == rg]
        wk = x.groupby(["date", "bin"]).x20.mean().unstack()                 # 주 × 구간
        out = {}
        for lab in LAB:
            s = wk[lab].dropna()
            rec, old = s[s.index >= RECENT], s[s.index < RECENT]
            xb = x[x.bin == lab]
            out[lab] = {"최근 x20 [CI]": [round(v * 100, 2) for v in boot(rec, rng)] if len(rec) > 12 else None,
                        "앞 기간": round(float(old.mean()) * 100, 2) if len(old) else None,
                        "반쪽": [round(float(rec[rec.index < HALF].mean()) * 100, 2), round(float(rec[rec.index >= HALF].mean()) * 100, 2)],
                        "최근 지수에 짐 비율": round(float((xb[xb.date >= RECENT].x20 < 0).mean()), 3),
                        "60일선 대비(중앙)": f"{xb.ma60.median():+.1%}"}
        res[rg] = {"주 수 최근/앞": [int((wk.index >= RECENT).sum()), int((wk.index < RECENT).sum())], "구간": out}
    sw = res["횡보·전환"]["구간"]
    start = None
    for i in range(len(LAB)):
        ok = all(sw[l]["최근 x20 [CI]"] and sw[l]["최근 x20 [CI]"][0] < 0 and max(sw[l]["반쪽"]) < 0 for l in LAB[i:])
        if ok:
            start = LAB[i]
            break
    res["판정"] = {"되돌림 시작 구간": start,
                 "최근경향": bool(start and sw[start]["최근 x20 [CI]"][2] < 0)}
    (results_dir() / "ma60_threshold.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
