"""수급을 길게(20 · 60 · 120일) — 프로그램 · 외국인 · 기관(금융투자 제외) — 검증이력 9.82 사전 등록 그대로.

  python research/validate_holding_flows.py   → results/holding_flows_validation.json
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
from universe_all import validation_codes
from validate_compression_regime import regime
from validate_holding_factors import boot, sample_days, spreads
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT_P, SPLIT_I, MIN_SPREAD = pd.Timestamp("2021-06-01"), pd.Timestamp("2024-03-01"), 0.01
WINS = (20, 60, 120)
REGS = ("상승장", "횡보·전환", "하락장")


def rows(code: str, days: pd.DatetimeIndex) -> pd.DataFrame | None:
    b = load_bars(LONG_HISTORY, code)
    prog = load_program(code)
    if b is None or len(b) < 300 or prog is None or prog.empty:
        return None
    b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    c = b["close"]
    pr = prog.reindex(b.index)
    f = pd.DataFrame(index=b.index)
    inv = load_detail(code)
    iv = inv.reindex(b.index) if inv is not None and not inv.empty else None
    for n in WINS:
        val = pr["value"].rolling(n, min_periods=int(n * 0.75)).sum()
        f[f"prog{n}"] = pr["prog_net"].rolling(n, min_periods=int(n * 0.75)).sum() / val
        if iv is not None:
            f[f"for{n}"] = iv["외국인"].rolling(n, min_periods=int(n * 0.75)).sum() * 1e6 / val        # 백만원 → 원
            f[f"inst{n}"] = (iv["기관합계"] - iv["금융투자"]).rolling(n, min_periods=int(n * 0.75)).sum() * 1e6 / val
    for h in (20, 60):
        f[f"fwd{h}"] = c.shift(-h) / c - 1
    keep = b.index.intersection(days)
    out = f.loc[keep].replace([np.inf, -np.inf], np.nan)
    out["code"] = code
    return out


def judge(sp: pd.DataFrame, split, rng) -> dict:
    a, b = sp[sp.index < split].spread, sp[sp.index >= split].spread
    if len(a) < 20 or len(b) < 20:
        return {"표본 부족": [len(a), len(b)]}
    f, t = boot(a, rng), boot(b, rng)
    excl = lambda x: x[1] > 0 if x[0] > 0 else x[2] < 0
    same = np.sign(f[0]) == np.sign(t[0])
    v = ("판단에 씀" if same and excl(f) and excl(t) and min(abs(f[0]), abs(t[0])) >= MIN_SPREAD else "참고(약함)" if same else "판단 없음")
    return {"앞": [round(x, 4) for x in f], "뒤": [round(x, 4) for x in t], "주 수": [len(a), len(b)], "verdict": v}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    days = sample_days()
    d = pd.concat([r for code in validation_codes() if (r := rows(code, days)) is not None])
    d.index.name = "date"
    for h in (20, 60):
        d[f"x{h}"] = d[f"fwd{h}"] - d.groupby(level=0)[f"fwd{h}"].transform("mean")
    d["reg"] = regime().reindex(d.index).to_numpy()
    res = {}
    for who, split in (("프로그램", SPLIT_P), ("외국인", SPLIT_I), ("기관(금융투자 제외)", SPLIT_I)):
        col0 = {"프로그램": "prog", "외국인": "for", "기관(금융투자 제외)": "inst"}[who]
        out = {}
        for n in WINS:
            col = f"{col0}{n}"
            if col not in d:
                continue
            x = d[d[col].notna()]
            r = {"종목": int(x.code.nunique())}
            for h in (20, 60):
                r[f"전체 {h}일"] = judge(spreads(x, col, f"x{h}"), split, rng)
            for g in REGS:
                xr = x[x.reg == g]
                r[f"{g} 20일"] = judge(spreads(xr, col, "x20"), split, rng) if len(xr) else None
            out[f"{n}일 누적"] = r
        res[who] = out
    (results_dir() / "holding_flows_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
