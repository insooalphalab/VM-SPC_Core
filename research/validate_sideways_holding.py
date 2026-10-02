"""횡보·전환 장세의 보유 판단 — '최근경향' 기준으로 다시 — 검증이력 9.86 사전 등록 그대로.

  python research/validate_sideways_holding.py   → results/sideways_holding_validation.json
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

from collect_investor_detail import load_detail
from collect_program import load_program
from validate_compression_regime import regime
from validate_holding_factors import FACTORS, boot, spreads
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

RECENT, HALF, T = pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01"), 0.01
WINS = (20, 60, 120)


def flow_cols(panel: pd.DataFrame) -> pd.DataFrame:
    """9.82 정의(프로그램 · 외국인 · 기관 − 금융투자, N일 누적 ÷ 거래대금)를 패널 날짜에 붙인다."""
    days = panel.index.unique()
    parts = []
    for code in panel.code.unique():
        b, prog = load_bars(LONG_HISTORY, code), load_program(code)
        if b is None or prog is None or prog.empty:
            continue
        pr = prog.reindex(b.index)
        inv = load_detail(code)
        iv = inv.reindex(b.index) if inv is not None and not inv.empty else None
        f = pd.DataFrame(index=b.index)
        for n in WINS:
            val = pr["value"].rolling(n, min_periods=int(n * 0.75)).sum()
            f[f"prog{n}w"] = pr["prog_net"].rolling(n, min_periods=int(n * 0.75)).sum() / val
            if iv is not None:
                f[f"for{n}w"] = iv["외국인"].rolling(n, min_periods=int(n * 0.75)).sum() * 1e6 / val
                f[f"inst{n}w"] = (iv["기관합계"] - iv["금융투자"]).rolling(n, min_periods=int(n * 0.75)).sum() * 1e6 / val
        f = f.loc[f.index.intersection(days)].replace([np.inf, -np.inf], np.nan)
        f["code"] = code
        parts.append(f)
    return pd.concat(parts)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    pnl = pd.read_csv(results_dir() / "holding_factors_panel.csv", parse_dates=["date"]).set_index("date")
    pnl["trend"] = pnl[["ma60", "ma120", "hi52", "rs"]].groupby(level=0).rank(pct=True).mean(axis=1, skipna=False)
    fl = flow_cols(pnl)
    pnl = pnl.reset_index().merge(fl.reset_index().rename(columns={"index": "date"}), on=["date", "code"], how="left").set_index("date")
    pnl["reg"] = regime().reindex(pnl.index).to_numpy()
    side = pnl[pnl.reg == "횡보·전환"]
    cands = {**FACTORS, "추세 강도 점수": "trend"}
    for n in WINS:
        cands |= {f"프로그램 {n}일": f"prog{n}w", f"외국인 {n}일": f"for{n}w", f"기관(금융투자 제외) {n}일": f"inst{n}w"}
    res, hits = {"횡보 주(앞/최근)": [int(side[side.index < RECENT].index.nunique()), int(side[side.index >= RECENT].index.nunique())]}, []
    for name, col in cands.items():
        x = side[side[col].notna()]
        x = x[x.groupby(level=0)[col].transform("size") >= 50]
        try:
            sp = spreads(x, col, "x20").spread                      # 하루 100종목 미만인 날은 spreads 가 건너뜀
        except KeyError:
            sp = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        rec = sp[sp.index >= RECENT]
        if len(rec) < 20:
            res[name] = {"표본 부족": int(len(rec))}
            continue
        old = sp[sp.index < RECENT]
        r = boot(rec, rng)
        h1, h2 = float(rec[rec.index < HALF].mean()), float(rec[rec.index >= HALF].mean())
        excl = lambda c: c[1] > 0 if c[0] > 0 else c[2] < 0
        recent_ok = abs(r[0]) >= T and excl(r) and np.sign(h1) == np.sign(h2) == np.sign(r[0]) and min(abs(h1), abs(h2)) >= T / 2
        o = boot(old, rng) if len(old) >= 20 else None
        verified = recent_ok and o is not None and np.sign(o[0]) == np.sign(r[0]) and excl(o) and abs(o[0]) >= T
        res[name] = {"앞 기간": None if o is None else [round(v, 4) for v in o], "최근 [CI]": [round(v, 4) for v in r],
                     "2021-06~2023 / 2024~": [round(h1, 4), round(h2, 4)], "주 수(앞/최근)": [int(len(old)), int(len(rec))],
                     "판정": "검증됨" if verified else "최근경향" if recent_ok else "아님"}
        if recent_ok:
            hits.append(name)
    res["걸린 요소"] = hits
    (results_dir() / "sideways_holding_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        if isinstance(v, dict) and "최근 [CI]" in v:
            f = lambda a: f"{a[0]*100:+.2f}[{a[1]*100:+.1f},{a[2]*100:+.1f}]"
            print(f"{k:18s} 앞 {f(v['앞 기간']) if v['앞 기간'] else '-':22s} 최근 {f(v['최근 [CI]']):22s} 반쪽 {v['2021-06~2023 / 2024~'][0]*100:+.2f}/{v['2021-06~2023 / 2024~'][1]*100:+.2f} {v['주 수(앞/최근)']} {v['판정']}")
        else:
            print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
