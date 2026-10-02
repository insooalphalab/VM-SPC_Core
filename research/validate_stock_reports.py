"""종목 리포트 — 60일 안 3개사 이상 목표가 상향이면 이후 지수를 이기는가 — 검증이력 9.106 사전 등록 그대로.

  python research/validate_stock_reports.py   → results/stock_reports_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "stock_track", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

from reclassify_recent import level_ci
from sector import load_reports
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

WIN, NEED = pd.Timedelta(days=60), 3
HALF = pd.Timestamp("2026-02-01")


def events(rep: pd.DataFrame, direction: str) -> pd.DataFrame:
    """종목별로 60일 안 서로 다른 증권사 direction 수가 처음 3곳 이상이 된 날."""
    out = []
    r = rep[rep.dir == direction].sort_values("date")
    for code, g in r.groupby("code"):
        last_hit = None
        dates, brokers = g.date.to_numpy(), g.broker.to_numpy()
        for i in range(len(g)):
            d = dates[i]
            w = (dates > d - np.timedelta64(60, "D")) & (dates <= d)
            n = len(set(brokers[w]))
            if n >= NEED and (last_hit is None or d - last_hit > np.timedelta64(60, "D")):
                out.append({"code": code, "date": pd.Timestamp(d), "brokers": n})
                last_hit = d
    return pd.DataFrame(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    rep = load_reports()
    rep = rep[rep.dir.isin(["up", "down"]) & rep.broker.notna()]
    idx = load_bars(LONG_HISTORY, "069500")
    res, cache = {}, {}
    for direction, lab in (("up", "U3 상향 3개사"), ("down", "D3 하향 3개사")):
        ev = events(rep, direction)
        rows = []
        for _, x in ev.iterrows():
            if x.code not in cache:
                b = load_bars(LONG_HISTORY, x.code)
                cache[x.code] = None if b is None or len(b) < 100 else b[(b[["open", "close"]] > 0).all(1)]
            b = cache[x.code]
            if b is None:
                continue
            e = int(b.index.searchsorted(x.date + pd.Timedelta(days=1)))   # 리포트 날 다음 거래일 시가
            if e >= len(b):
                continue
            row = {"code": x.code, "entry_date": b.index[e]}
            io = idx["open"].reindex(b.index).to_numpy(float)
            ic = idx["close"].reindex(b.index).to_numpy(float)
            o, c = b["open"].to_numpy(float), b["close"].to_numpy(float)
            for k in (20, 60):
                j = e + k - 1
                row[f"x{k}"] = ((c[j] / o[e] - 1) - (ic[j] / io[e] - 1)) * 100 if j < len(c) and np.isfinite(ic[j]) else np.nan
            rows.append(row)
        d = pd.DataFrame(rows)
        d.to_csv(results_dir() / f"stock_reports_{direction}_events.csv", index=False)
        x = d.dropna(subset=["x20"]).assign(R=lambda z: z.x20)
        ci = level_ci(x, rng) if len(x) >= 30 else (np.nan,) * 3
        h1, h2 = float(x[x.entry_date < HALF].R.mean()), float(x[x.entry_date >= HALF].R.mean())
        if direction == "up":
            ok = ci[0] >= 1.5 and ci[1] > 0 and min(h1, h2) >= 0.75
        else:
            ok = ci[0] <= -1.5 and ci[2] < 0 and max(h1, h2) <= -0.75
        x60 = d.x60.dropna()
        res[lab] = {"사건(일봉 있는 종목)": len(d), "20일 계산 가능": len(x), "20일 초과 %p [CI]": [round(q, 2) for q in ci],
                    "반쪽(2026-02 전/후)": [round(h1, 2), round(h2, 2)], "지수 이긴 비율": round(float((x.R > 0).mean()), 3),
                    "60일 초과 %p(서술)": [round(float(x60.mean()), 2), len(x60)], "판정": "잠정 통과" if ok else "아님"}
    (results_dir() / "stock_reports_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
