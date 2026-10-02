"""장세별 보유 판단 — 추세 강도 점수 × 장세 — 검증이력 9.81 사전 등록 그대로. 자료 = 9.80 패널.

  python research/validate_holding_regime.py   → results/holding_regime_validation.json
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

from validate_compression_regime import regime
from validate_holding_factors import boot, spreads
from v2_config import results_dir

SPLIT, MIN_SPREAD = pd.Timestamp("2021-06-01"), 0.01
TREND = ("ma60", "ma120", "hi52", "rs")
REGS = ("상승장", "횡보·전환", "하락장")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "holding_factors_panel.csv", parse_dates=["date"]).set_index("date")
    d["reg"] = regime().reindex(d.index).to_numpy()
    d["trend"] = d[list(TREND)].groupby(level=0).rank(pct=True).mean(axis=1, skipna=False)
    res = {}
    for name, col in (("추세 강도 점수", "trend"), ("볼린저 폭(서술)", "bbw"), ("위쪽 매물 비중(서술)", "vp")):
        out = {}
        for r in REGS:
            x = d[d.reg == r]
            row = {"날짜": int(x.index.nunique())}
            for h in (20, 60):
                sp = spreads(x, col, f"x{h}")
                f, t = boot(sp[sp.index < SPLIT].spread, rng), boot(sp[sp.index >= SPLIT].spread, rng)
                row[f"{h}일 맞춤"], row[f"{h}일 예측"] = [round(v, 4) for v in f], [round(v, 4) for v in t]
            f, t = row["20일 맞춤"], row["20일 예측"]
            excl = lambda c: c[1] > 0 if c[0] > 0 else c[2] < 0
            same = np.sign(f[0]) == np.sign(t[0])
            if col == "trend":
                row["verdict"] = ("장세별 판단에 씀" if same and excl(f) and excl(t) and min(abs(f[0]), abs(t[0])) >= MIN_SPREAD
                                  else "참고(약함)" if same else "판단 없음")
                row["방향"] = "강한 종목 보유 유리" if t[0] > 0 else "강한 종목 이후 약함"
            out[r] = row
        res[name] = out
    (results_dir() / "holding_regime_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
