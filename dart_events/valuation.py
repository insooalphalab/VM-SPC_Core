"""C. 바스켓 밸류에이션 상태 (검증이력 9.17) — 예측이 아니라 "지금 자기 5년 범위에서 비싼가 싼가".

  바스켓 PER = Σ시가총액 ÷ Σ최근 4분기 당기순이익, PBR = Σ시가총액 ÷ Σ자본총계
  시가총액 ≈ 수정 종가 × 현재 상장주식수, 재무는 접수일부터 반영(point-in-time).
  python dart_events/valuation.py   → results/valuation_state.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "dart_events"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys

import numpy as np
import pandas as pd

from client import dart_dir
from fundamentals import quarterly
from v2_config import load_baskets, results_dir
from v2_datastore import load_bars

MIN_COVER = 0.7        # 바스켓 종목 중 재무·주식수가 있는 비율이 이보다 낮은 날은 계산하지 않음


def _asof(q: pd.DataFrame, col: str, index: pd.DatetimeIndex) -> pd.Series:
    s = q.dropna(subset=[col]).assign(d=lambda x: pd.to_datetime(x["rcept_dt"], format="%Y%m%d"))
    s = s.sort_values("d").drop_duplicates("d", keep="last").set_index("d")[col]
    return s.reindex(index.union(s.index)).ffill().reindex(index)


def stock_fundamentals(fin: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ni, eq = quarterly(fin, "ni"), quarterly(fin, "equity")
    out = {}
    for code, g in ni.groupby("stock_code"):
        g = g.set_index(["year", "q"]).sort_index()
        full = pd.MultiIndex.from_product([range(g.index.get_level_values(0).min(), g.index.get_level_values(0).max() + 1), (1, 2, 3, 4)])
        g = g.reindex(full)
        g["ttm"] = g["value"].rolling(4, min_periods=4).sum()
        out[code] = {"ni": g.dropna(subset=["rcept_dt"]).reset_index(drop=True)}
    for code, g in eq.groupby("stock_code"):
        out.setdefault(code, {})["eq"] = g.rename(columns={"value": "equity"})
    return out


def basket_series(basket: dict, funda: dict, shares: dict) -> pd.DataFrame | None:
    mcap, ni, eq = {}, {}, {}
    for s in basket["sensors"]:
        c = s["code"]
        bars, f = load_bars(basket["name"], c), funda.get(c, {})
        if bars is None or c not in shares or "ni" not in f or "eq" not in f:
            continue
        mcap[c] = bars["close"] * shares[c]
        ni[c] = _asof(f["ni"], "ttm", bars.index)
        eq[c] = _asof(f["eq"], "equity", bars.index)
    if not mcap:
        return None
    M, N, E = (pd.DataFrame(x).sort_index() for x in (mcap, ni, eq))
    ok = M.notna() & N.notna() & E.notna()
    cover = ok.sum(1) / len(basket["sensors"])
    tot_m, tot_n, tot_e = M.where(ok).sum(1), N.where(ok).sum(1), E.where(ok).sum(1)
    df = pd.DataFrame({"per": np.where(tot_n > 0, tot_m / tot_n, np.nan), "pbr": tot_m / tot_e.replace(0, np.nan),
                       "cover": cover}, index=M.index)
    return df[df["cover"] >= MIN_COVER]


def pct_rank(s: pd.Series) -> float | None:
    s = s.dropna()
    return None if len(s) < 250 else round(float((s <= s.iloc[-1]).mean()), 3)


def run_all() -> dict:
    fin = pd.read_csv(dart_dir() / "financials.csv", dtype={"stock_code": str, "rcept_dt": str})
    shares = pd.read_csv(dart_dir() / "shares.csv", dtype={"stock_code": str}).set_index("stock_code")["shares"].to_dict()
    funda = stock_fundamentals(fin)
    out = {}
    for b in load_baskets():
        df = basket_series(b, funda, shares)
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        out[b["name"]] = {"as_of": df.index[-1].strftime("%Y-%m-%d"),
                          "per": None if pd.isna(last["per"]) else round(float(last["per"]), 1),
                          "pbr": None if pd.isna(last["pbr"]) else round(float(last["pbr"]), 2),
                          "per_pct": pct_rank(df["per"]) if pd.notna(last["per"]) else None,
                          "pbr_pct": pct_rank(df["pbr"]), "cover": round(float(last["cover"]), 2),
                          "since": df.index[0].strftime("%Y-%m-%d")}
    (results_dir() / "valuation_state.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_state() -> dict:
    p = results_dir() / "valuation_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for k, v in run_all().items():
        print(f"{k:40s} PER {v['per']} ({v['per_pct']})  PBR {v['pbr']} ({v['pbr_pct']})  cover {v['cover']}")
