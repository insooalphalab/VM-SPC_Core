"""DART 분기 재무 → 돌파일 시점의 실적 성장 표시(검증이력 9.78). 그날까지 접수된 가장 최근 분기만 쓴다(미래 정보 없음).

  c_ni  = 최근 분기 순이익 > 0, 전년 같은 분기 > 0, 증가율 ≥ 25%
  c_rev = 최근 분기 매출 전년 같은 분기 대비 ≥ 25%
  a_ni  = 최근 4개 분기 순이익 합이 그 전 4개 분기 합(둘 다 > 0) 대비 ≥ 20%
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from client import dart_dir
from fundamentals import quarterly

Q_GROW, REV_GROW, A_GROW, FRESH = 0.25, 0.25, 0.20, 20


def table(fin: pd.DataFrame, account: str) -> dict[str, pd.DataFrame]:
    q = quarterly(fin, account)
    q["rcept"] = pd.to_datetime(q["rcept_dt"], format="%Y%m%d", errors="coerce")
    q["k"] = q["year"] * 4 + q["q"]
    return {c: g.set_index("k").sort_index() for c, g in q.groupby("stock_code")}


def load_tables() -> tuple[dict, dict]:
    fin = pd.read_csv(dart_dir() / "financials.csv", dtype={"stock_code": str, "rcept_dt": str})
    return table(fin, "ni"), table(fin, "rev")


def features(code: str, day: pd.Timestamp, ni: dict, rev: dict) -> dict:
    """day 전날까지 접수된 분기 기준. 값: 1.0 충족 · 0.0 미충족 · nan 자료 없음. rev_g·a_g = 증가율(표시용)."""
    out = {"c_ni": np.nan, "c_rev": np.nan, "a_ni": np.nan, "fresh": np.nan, "has": False, "rev_g": np.nan, "a_g": np.nan}
    g = ni.get(code)
    if g is None:
        return out
    known = g[g.rcept < day]
    if known.empty:
        return out
    k = int(known.index.max())
    out["has"] = True
    out["fresh"] = float((day - known.at[k, "rcept"]).days <= FRESH)
    v = known["value"]
    if k - 4 in v.index:
        cur, prev = v.get(k), v.get(k - 4)
        out["c_ni"] = float(cur > 0 and prev > 0 and cur / prev - 1 >= Q_GROW)
    last4, prev4 = [v.get(k - i) for i in range(4)], [v.get(k - 4 - i) for i in range(4)]
    if all(x is not None and pd.notna(x) for x in last4 + prev4):
        a, b = sum(last4), sum(prev4)
        out["a_ni"] = float(a > 0 and b > 0 and a / b - 1 >= A_GROW)
        if a > 0 and b > 0:
            out["a_g"] = a / b - 1
    r = rev.get(code)
    if r is not None:
        rk = r[r.rcept < day]["value"]
        if k in rk.index and k - 4 in rk.index and rk.get(k - 4) and rk.get(k - 4) > 0:
            out["rev_g"] = rk.get(k) / rk.get(k - 4) - 1
            out["c_rev"] = float(out["rev_g"] >= REV_GROW)
    return out
