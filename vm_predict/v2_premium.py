"""ETF 괴리율(시장가 vs NAV) → ETF 자체 예측의 보조 입력.

괴리율은 매일 거의 0 으로 되돌아가므로, 오늘 NAV 보다 비싸게(프리미엄) 끝난 ETF 는 그 되돌림만큼
다음 날 종가 수익률이 깎이고 할인이면 더해진다. 2026-09-27 39개 타겟(중복 제외 36개 ETF) 백필 분석:
ETF의 92%에서 오늘 괴리율과 다음 날 수익률이 음의 상관, 평소보다 할인된 날(dz<-1) 다음 날 상승
55.0% vs 프리미엄(dz>1) 48.4%. breadth 예측과 같은 방향일 때 적중 52.3% vs 반대 47.8%(차이 95% CI
[+1.6, +7.5]%p, 날짜 클러스터 부트스트랩) — 그래서 예측 방향(breadth, 임계값 고정)은 그대로 두고
"오늘 괴리율이 예측과 같은 편인가"에 따라 적중률을 보정한다.

보정은 **전 ETF 합산 효과**로 한다(v2_etf_extras.premium_adjustment): 그 ETF의 같은 방향 예측 적중률
+ (전체 풀에서 그 괴리율 상태의 적중률 - 전체 적중률). 처음엔 ETF별로 같은 조건인 날만 추려 적중률을
냈는데, 한 ETF를 3등분하면 조건당 약 40일이라 우연 변동이 효과(+6%p)보다 컸다 — KODEX 반도체는 동의
69.2% vs 반대 68.4%로 차이가 없는데도 "69%"로 올라 보였다(2026-09-28 사용자 지적으로 확인).

  dz = (오늘 괴리율 - 과거 60일 평균) / 과거 60일 표준편차   (인과적, 오늘 포함·미래 미사용)
  |dz| < NEUTRAL_BAND  → 중립(정보 없음)
  상승 예측 & 할인(dz<0), 하락 예측 & 프리미엄(dz>0) → 동의, 그 반대 → 반대
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

from v2_datastore import load_nav

DZ_WINDOW = 60
DZ_MIN_PERIODS = 40
NEUTRAL_BAND = 0.5      # 사전 고정값 — 분석은 부호(0)와 |dz|≥1 두 기준 모두에서 같은 방향이었다

AGREE, NEUTRAL, OPPOSE = "동의", "중립", "반대"


def premium_series(basket_name: str, target_code: str) -> pd.DataFrame | None:
    nav = load_nav(basket_name, target_code)
    if nav is None or nav.empty:
        return None
    df = nav[["dprt"]].copy()
    m = df["dprt"].rolling(DZ_WINDOW, min_periods=DZ_MIN_PERIODS).mean()
    s = df["dprt"].rolling(DZ_WINDOW, min_periods=DZ_MIN_PERIODS).std()
    df["dz"] = (df["dprt"] - m) / s.replace(0, np.nan)
    return df


def agreement(pred_up: pd.Series, dz: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pred_up.index, dtype=object)
    valid = pred_up.notna() & dz.notna()
    up = pred_up.astype("boolean")
    neutral = valid & (dz.abs() < NEUTRAL_BAND)
    agree = valid & ~neutral & ((up & (dz < 0)) | (~up & (dz > 0)))
    oppose = valid & ~neutral & ~agree
    out[neutral], out[agree], out[oppose] = NEUTRAL, AGREE, OPPOSE
    return out
