"""애널리스트 리포트 순상향 → 섹터 ETF 초과수익 (검증이력 9.25 사전 등록 기준 그대로).

  python research/validate_report_sector.py     → results/report_sector_validation.json
먼저: python reports/collect_web.py && python reports/parse.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "dart_events",
           _ROOT / "stock_track", _ROOT / "scenario", _ROOT / "research", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys

import numpy as np
import pandas as pd

from sector import BENCH, load_reports, sectors, signal
from v2_config import results_dir
from v2_datastore import load_bars

HORIZONS = (5, 20, 60)
MAIN_H = 20
MIN_SECTORS = 6
N_BOOT = 2000


def etf_closes() -> pd.DataFrame:
    return pd.DataFrame({name: load_bars(name, s["etf_code"])["close"] for name, s in sectors().items()
                         if load_bars(name, s["etf_code"]) is not None}).sort_index()


def evaluate(net: pd.DataFrame, px: pd.DataFrame, bm: pd.Series, H: int, rng) -> dict:
    fwd = (px.shift(-1 - H) / px.shift(-1) - 1).sub(bm.shift(-1 - H) / bm.shift(-1) - 1, axis=0)   # 다음 날 종가 진입
    ics, spr = [], []
    for d in net.index:
        s, f = net.loc[d], fwd.loc[d] if d in fwd.index else None
        if f is None:
            continue
        ok = s.notna() & f.notna()
        if ok.sum() < MIN_SECTORS:
            continue
        s, f = s[ok], f[ok]
        ics.append((d, s.rank().corr(f.rank())))
        k = max(1, len(s) // 3)
        o = s.sort_values().index
        spr.append((d, f[o[-k:]].mean() - f[o[:k]].mean()))
    if not ics:
        return {"H": H, "n_days": 0, "status": "데이터 없음"}
    ic, sp = pd.Series(dict(ics)), pd.Series(dict(spr))
    blocks = np.array([ic.iloc[i:i + H].mean() for i in range(0, len(ic), H)])
    bsp = np.array([sp.iloc[i:i + H].mean() for i in range(0, len(sp), H)])
    idx = rng.integers(0, len(blocks), (N_BOOT, len(blocks)))
    lo, hi = np.percentile(blocks[idx].mean(1), [2.5, 97.5])
    slo, shi = np.percentile(bsp[idx].mean(1), [2.5, 97.5])
    passed = lo > 0
    status = ("잠정 통과" if passed else "HOLD") if H == MAIN_H else ("참고(기준 충족)" if passed else "참고")
    return {"H": H, "n_days": int(len(ic)), "n_blocks": int(len(blocks)), "first": str(ic.index[0].date()),
            "last": str(ic.index[-1].date()), "ic": round(float(ic.mean()), 4), "ic_ci": [round(lo, 4), round(hi, 4)],
            "spread": round(float(sp.mean()), 4), "spread_ci": [round(slo, 4), round(shi, 4)], "status": status}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    rep = load_reports()
    if rep.empty:
        print("리포트 표 없음 — reports/collect_web.py, reports/parse.py 먼저")
        return 2
    px = etf_closes()
    bm = load_bars("kospi_top10_to_etf", BENCH)["close"].reindex(px.index)
    net, n = signal(rep, px.index)
    res = [evaluate(net, px, bm, H, rng) for H in HORIZONS]
    cover = {name: int(n[name].iloc[-1]) for name in n.columns}
    out = {"n_reports": int(len(rep)), "report_period": [str(rep["date"].min().date()), str(rep["date"].max().date())],
           "n_sectors": int(net.shape[1]), "reports_last20_by_sector": cover, "results": res}
    (results_dir() / "report_sector_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
