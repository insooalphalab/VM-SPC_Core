"""리포트 표(data/_reports/reports.csv) → 섹터(테마 대표 바스켓)별 리포트 흐름 (검증이력 9.25 정의).

섹터 = basket_watchlist 의 `*_to_etf` 바스켓(센서 종목 → 타겟 ETF). KODEX 200 이 타겟인 바스켓은 기준 지수라 제외.
날짜 d 의 신호 = 작성일 기준 직전 WINDOW 거래일(d 포함) 리포트의 순상향 비율 (상향 − 하향) ÷ 전체, 리포트 MIN_REPORTS 건 미만은 NaN.
휴장일에 작성된 리포트는 다음 거래일로 센다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

from v2_config import data_dir, load_baskets

WINDOW = 20
MIN_REPORTS = 5
BENCH = "069500"


def sectors() -> dict[str, dict]:
    out = {}
    for b in load_baskets():
        if b["name"].endswith("_to_etf") and b["target"]["code"] != BENCH:
            out[b["name"]] = {"etf_code": b["target"]["code"], "etf_name": b["target"]["name"],
                              "codes": {s["code"] for s in b["sensors"]}}
    return out


def load_reports() -> pd.DataFrame:
    p = data_dir() / "_reports" / "reports.csv"
    df = pd.read_csv(p, dtype={"code": str}, parse_dates=["date"]) if p.exists() else pd.DataFrame()
    return df


def counts(reports: pd.DataFrame, cal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """섹터별 거래일 × {up, down, flat, new, n} 일별 건수."""
    secs = sectors()
    r = reports.copy()
    r["ti"] = cal.searchsorted(r["date"])                 # 휴장일 작성 → 다음 거래일
    r = r[r["ti"] < len(cal)]
    out = {}
    for name, s in secs.items():
        g = r[r["code"].isin(s["codes"])]
        m = pd.DataFrame(0, index=range(len(cal)), columns=["up", "down", "flat", "new", "n"])
        for k in ("up", "down", "flat", "new"):
            vc = g[g["dir"] == k]["ti"].value_counts()
            m.loc[vc.index, k] = vc.values
        vc = g[g["dir"] != "none"]["ti"].value_counts()
        m.loc[vc.index, "n"] = vc.values
        m.index = cal
        out[name] = m
    return out


def signal(reports: pd.DataFrame, cal: pd.DatetimeIndex, window: int = WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(순상향 비율, 리포트 수) — 행 = 거래일, 열 = 섹터. 리포트 데이터 시작 전 구간은 NaN."""
    c = counts(reports, cal)
    start = cal.searchsorted(reports["date"].min()) + window - 1
    net, n = {}, {}
    for name, m in c.items():
        roll = m.rolling(window, min_periods=window).sum()
        cnt = roll["n"]
        net[name] = ((roll["up"] - roll["down"]) / cnt).where(cnt >= MIN_REPORTS)
        n[name] = cnt
    net, n = pd.DataFrame(net), pd.DataFrame(n)
    net.iloc[:start] = np.nan
    return net, n
