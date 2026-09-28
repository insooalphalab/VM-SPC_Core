"""ETF 자체 예측의 신뢰도 지수 — 방향을 새로 예측하지 않고 "오늘 이 예측을 믿어도 되는가"만 판정한다.

가상계측(VM)의 신뢰도 지수(RI/GSI) 개념을 가져와 세 가지를 본다. 전부 인과적(t 시점에 알 수 있는
값만 사용):

  1. 유사도   — 오늘 Hotelling T²(바스켓 신호가 평소 분포에서 떨어진 정도)가 직전 250일 중 몇 백분위인가.
  2. 최근 성능 — 예측기 자신의 적중/실패에 건 베르누이 CUSUM. 직전 250건 적중률(p0)에서 10%p 나빠지는
                변화를 잡는다. 관리도를 주가가 아니라 예측기에 거는 구조.
  3. 변동성 국면 — 타겟 ETF 20일 실현변동성이 직전 250일 중 몇 백분위인가.

등급은 학습 모델이 아니라 사전에 고정한 규칙으로 합친다(과적합 층을 하나 더 얹지 않기 위해):
  낮음 = T² 95백분위 초과 or CUSUM 경보 or 변동성 90백분위 초과
  높음 = T² 80백분위 미만 and 경보 없음 and 변동성 70백분위 미만
  보통 = 나머지
이 등급이 실제로 적중률을 가르는지는 v2_etf_extras.py 가 전 바스켓 풀링으로 검증하고, 검증을 통과했을
때만 화면에 표시한다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LOOKBACK = 250
MIN_LOOKBACK = 120
VOL_WINDOW = 20
CUSUM_SHIFT = 0.10
CUSUM_H = 2.5

HIGH, MID, LOW = "높음", "보통", "낮음"


def _trailing_pct(s: pd.Series) -> pd.Series:
    """오늘 값이 직전 LOOKBACK 개(오늘 제외) 중 몇 백분위인가."""
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            continue
        past = vals[max(0, i - LOOKBACK):i]
        past = past[~np.isnan(past)]
        if len(past) >= MIN_LOOKBACK:
            out[i] = (past < vals[i]).mean()
    return pd.Series(out, index=s.index)


def _hit_cusum(hit: pd.Series) -> pd.Series:
    """t 시점 통계량은 t-1 까지 결과가 확정된 예측만 쓴다(t 의 예측 결과는 t+1 종가에 확정)."""
    h = hit.to_numpy(dtype=float)
    stat = np.full(len(h), np.nan)
    s = 0.0
    history: list[float] = []
    for i in range(len(h)):
        if len(history) >= MIN_LOOKBACK // 2:
            stat[i] = s
        if np.isnan(h[i]):
            continue
        p0 = min(max(np.mean(history[-LOOKBACK:]) if history else 0.5, 0.5), 0.8)
        q0 = 1 - p0
        q1 = min(q0 + CUSUM_SHIFT, 0.95)
        s = max(0.0, s + (math.log(q1 / q0) if h[i] == 0 else math.log((1 - q1) / (1 - q0))))
        history.append(h[i])
    return pd.Series(stat, index=hit.index)


def reliability_frame(close: pd.Series, t2: pd.Series, hit: pd.Series) -> pd.DataFrame:
    idx = close.index
    vol = np.log(close).diff().rolling(VOL_WINDOW).std()
    df = pd.DataFrame({
        "t2_pct": _trailing_pct(t2.reindex(idx)),
        "vol_pct": _trailing_pct(vol),
        "cusum": _hit_cusum(hit.reindex(idx)),
    }, index=idx)
    df["cusum_alarm"] = df["cusum"] > CUSUM_H
    ready = df[["t2_pct", "vol_pct", "cusum"]].notna().all(axis=1)
    low = (df["t2_pct"] > 0.95) | df["cusum_alarm"] | (df["vol_pct"] > 0.90)
    high = (df["t2_pct"] < 0.80) & ~df["cusum_alarm"] & (df["vol_pct"] < 0.70)
    grade = pd.Series(MID, index=idx, dtype=object)
    grade[low], grade[high & ~low] = LOW, HIGH
    grade[~ready] = None
    df["grade"] = grade
    return df


def reasons(row: pd.Series) -> list[str]:
    out = []
    if row["t2_pct"] > 0.95:
        out.append("바스켓 신호가 평소와 크게 다름")
    if row["cusum_alarm"]:
        out.append("최근 이 예측의 적중률이 떨어지는 중")
    if row["vol_pct"] > 0.90:
        out.append("변동성이 평소보다 매우 큼")
    return out
