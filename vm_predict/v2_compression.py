"""가격 응축 상태 — "곧 크게 움직일 가능성이 높다(방향은 모름)"를 알려주는 모니터링 지표.

응축 = 20일 실현변동성과 20일 고저폭(최고가−최저가)/종가가 모두 자기 직전 250일 중 하위 20%.

2026-09-28 검증(검증이력 9.14): 응축 뒤 20일 변동성이 오늘의 1.5배 이상으로 커진 비율이 ETF 27% vs 평소 14%,
종목 41% vs 17%로 뚜렷했다. 반면 응축 후 돌파한 날의 이후 유리폭/불리폭은 아무 날과 같아서 방향 정보는 없다.
그래서 이 지표는 "큰 움직임 경보"로만 쓰고 방향 예측에는 쓰지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 250
MIN_LOOKBACK = 120
PCT_CUT = 0.20
WINDOW = 20
EXPAND = 1.5


def _trailing_pct(s: pd.Series) -> pd.Series:
    v = s.to_numpy(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        past = v[max(0, i - LOOKBACK):i]
        past = past[~np.isnan(past)]
        if len(past) >= MIN_LOOKBACK:
            out[i] = (past < v[i]).mean()
    return pd.Series(out, index=s.index)


def compression_frame(bars: pd.DataFrame) -> pd.DataFrame:
    c, h, l = bars["close"], bars["high"], bars["low"]
    r = np.log(c).diff()
    rv = r.rolling(WINDOW).std()
    rng = (h.rolling(WINDOW).max() - l.rolling(WINDOW).min()) / c
    rv_pct, rng_pct = _trailing_pct(rv), _trailing_pct(rng)
    fut_rv = r[::-1].rolling(WINDOW).std()[::-1].shift(-1)   # t+1..t+20 실현변동성(검증용)
    return pd.DataFrame({"rv_pct": rv_pct, "rng_pct": rng_pct,
                         "compressed": (rv_pct <= PCT_CUT) & (rng_pct <= PCT_CUT),
                         "expanded": (fut_rv / rv >= EXPAND).where(fut_rv.notna() & rv.notna())})


def current_state(frame: pd.DataFrame) -> dict | None:
    f = frame.dropna(subset=["rv_pct", "rng_pct"])
    if f.empty:
        return None
    comp = f["compressed"].to_numpy()
    streak = 0
    for x in comp[::-1]:
        if not x:
            break
        streak += 1
    return {"date": f.index[-1].strftime("%Y-%m-%d"), "compressed": bool(comp[-1]), "streak": streak,
            "rv_pct": float(f["rv_pct"].iloc[-1]), "rng_pct": float(f["rng_pct"].iloc[-1])}


def pooled_expansion(frames: list[pd.DataFrame]) -> dict | None:
    """여러 ETF를 합쳐 '응축 뒤 20일 변동성 1.5배↑' 비율 vs 평소 비율."""
    d = pd.concat([f.dropna(subset=["expanded", "rv_pct", "rng_pct"]) for f in frames]) if frames else None
    if d is None or d.empty or not d["compressed"].any():
        return None
    return {"comp_rate": float(d.loc[d["compressed"], "expanded"].mean()), "base_rate": float(d["expanded"].mean()),
            "n_comp": int(d["compressed"].sum())}
