"""누적 수급 피처(검증이력 9.8 사전 등록 그대로).

종목별 투자자 매매동향을 data/_investor/{코드}.csv 에 종목당 한 번만 저장한다(여러 바스켓에 속해도 공유).
  frgn_flow5/20 : 외국인 5·20일 누적 순매수 ÷ 같은 기간 거래대금
  inst_flow5/20 : 기타기관(기관합계 − 금융투자) 동일
하루 늦춰(t-1 까지 누적) 쓴다 — 16:30 실행 시점엔 당일 수급이 확정되지 않았을 수 있어서.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from datetime import datetime, timedelta

import pandas as pd

from kis_client import RateLimitedCaller
from v2_config import KST, data_dir
from v2_investor_flow import FLOW_COLS as RAW_COLS, fetch_investor_daily

FLOW_FEATURES = ["frgn_flow5", "frgn_flow20", "inst_flow5", "inst_flow20"]
REFRESH_DAYS = 15


def flow_path(code: str) -> _Path:
    return data_dir() / "_investor" / f"{code}.csv"


def load_flow(code: str) -> pd.DataFrame | None:
    path = flow_path(code)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"date": str})
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()[RAW_COLS].astype(float)


def update_flow(caller: RateLimitedCaller, code: str, days: int) -> tuple[int, str]:
    """저장분이 있으면 최근 REFRESH_DAYS 부터(과거 구간이 모자라면 앞쪽도) 받아 병합."""
    today = datetime.now(KST).date()
    full_start = today - timedelta(days=days)
    existing = load_flow(code)
    frames = []
    if existing is None:
        frames.append(fetch_investor_daily(caller, code, full_start, today, max_pages=80))
        mode = "백필"
    else:
        frames.append(fetch_investor_daily(caller, code, existing.index.max().date() - timedelta(days=REFRESH_DAYS), today))
        if existing.index.min().date() > full_start + timedelta(days=7):
            frames.append(fetch_investor_daily(caller, code, full_start, existing.index.min().date() - timedelta(days=1),
                                               max_pages=80))
        mode = "증분"
    new = pd.concat(frames, ignore_index=True)
    path = flow_path(code)
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing is not None:
        old = pd.read_csv(path, dtype={"date": str})
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates("date", keep="last").sort_values("date")
    new.to_csv(path, index=False, encoding="utf-8-sig")
    return len(new), mode


def flow_features(code: str, bars: pd.DataFrame) -> pd.DataFrame | None:
    flow = load_flow(code)
    if flow is None:
        return None
    flow = flow.reindex(bars.index)
    value = bars["close"] * bars["volume"] / 1e6       # 거래대금(백만원) — 수급 단위와 맞춤
    other_inst = flow["기관합계"] - flow["금융투자"]
    out = pd.DataFrame(index=bars.index)
    for n in (5, 20):
        v = value.rolling(n).sum()
        out[f"frgn_flow{n}"] = flow["외국인"].rolling(n).sum() / v
        out[f"inst_flow{n}"] = other_inst.rolling(n).sum() / v
    return out.shift(1)[FLOW_FEATURES]
