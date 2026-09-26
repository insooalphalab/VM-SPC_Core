"""Layer 2 — VM-SPC Core 피처 5종 산출 (명세 2.2).

기존 core/ 로직(칼만·CUSUM·T²)을 그대로 재사용하는 wrapper 로만 구성한다:
  kalman_resid  = 칼만 1스텝 잔차 / 롤링σ (core.v2_signals.kalman_features 의 band_z)
  price_z       = 로그종가의 롤링 Z-Score (window=PRICE_Z_WINDOW, 당일 포함 — EOD 확정값이라 인과적)
  cusum         = 칼만 잔차 양방향 CUSUM 의 순누적합 C+ + C− (core.v2_cusum.cusum)
  hotelling_t2  = 종목별 [칼만잔차·기울기z·거래량z] 3변량 T² (core.v2_hotelling.hotelling_t2)
  coint_z       = 종목 vs 타겟 ETF 롤링 OLS 공적분 스프레드의 Z-Score

설계 판단(명세 2.11 미결정 사항에 대한 결정):
- T² 입력: vm_predict 의 바스켓 평균 T²는 매물대(profile, 250봉 워밍업)가 들어가 3년치 데이터에서
  유효구간이 1년 남짓밖에 안 남는다 — Walk-Forward fold 가 2개 이하로 줄어 검증 자체가 안 된다.
  그래서 함수(hotelling_t2)는 그대로 쓰고 입력만 칼만 파생 3변량(종목별)으로 바꿨다. 종목별 값이라
  풀링 패널에서 종목 간 변동도 생긴다.
- coint_z: pair_spc.analyze_pair 는 전체기간 OLS 한 번(사후 적합)이라 ML 피처로 쓰면 look-ahead 가
  생긴다. 같은 정의(log A − α − β·log B, 롤링 Z)를 유지하되 α·β 를 직전 COINT_OLS_WINDOW 일로만
  추정하는 인과적 버전으로 바꿨다. B 는 ETF-ex-A 가 아니라 타겟 ETF 그대로 쓴다 — ex-self 에 필요한
  ETF 내 편입비중은 Stage0 후보 5종목에만 있어서 바스켓 전 종목에 적용할 수 없다.
- 롤링 창 길이는 기존 스코어카드와 동일 창을 쓴다(σ·T²·SPC 모두 60일, price_z 는 rs_lookback 20일).
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

from v2_config import Params
from v2_cusum import cusum
from v2_hotelling import hotelling_t2
from v2_signals import kalman_features

FEATURE_COLS = ["kalman_resid", "price_z", "cusum", "hotelling_t2", "coint_z"]
FEATURE_LABELS = {"kalman_resid": "칼만 잔차", "price_z": "가격 Z-Score", "cusum": "CUSUM",
                  "hotelling_t2": "Hotelling T²", "coint_z": "공적분 잔차 Z"}

PRICE_Z_WINDOW = 20          # Params.rs_lookback 과 동일
COINT_OLS_WINDOW = 120       # 롤링 헤지비율 추정창(약 6개월)
COINT_Z_WINDOW = 60          # PairParams.spc_window 와 동일
T2_KEYS = ["band_z", "slope_z", "vol_z"]


def compute_kalman_residual(kf: pd.DataFrame) -> pd.Series:
    return kf["band_z"]


def compute_price_zscore(close: pd.Series, window: int = PRICE_Z_WINDOW) -> pd.Series:
    logc = np.log(close)
    mu = logc.rolling(window, min_periods=window).mean()
    sd = logc.rolling(window, min_periods=window).std()
    return (logc - mu) / sd


def compute_cusum(kf: pd.DataFrame, p: Params) -> pd.Series:
    res = cusum(kf["innov"], kf["sigma"], p.cusum_k, p.cusum_h)
    return res.c_pos + res.c_neg


def compute_hotelling_t2(kf: pd.DataFrame, p: Params) -> pd.Series:
    """log1p(T²) — T²는 카이제곱형이라 한쪽으로 극단적으로 길게 뻗는다(실측: kospi_top10_to_etf 에서
    중앙값 2.0 vs 최댓값 5870, 99.9백분위의 70배인 단일 관측치). Ridge 의 StandardScaler 는 이 하나의
    극단치가 표준편차를 통째로 부풀려 나머지 관측치를 0 근처로 짓누르고, 그 여파로 표준화계수가
    다른 피처(0.03~0.2대)보다 한 자릿수 큰 1.39까지 튀는 것을 실측으로 확인했다(y와의 상관은 0.017로
    사실상 없는데도). log1p(단조변환, LightGBM 분기순서엔 영향 없음)로 눌렀더니 계수가 0.09로
    정상화됐다."""
    return np.log1p(hotelling_t2(kf[T2_KEYS], p.t2_window, p.t2_alpha).t2)


def compute_cointegration_residual_z(close_a: pd.Series, close_b: pd.Series,
                                     ols_window: int = COINT_OLS_WINDOW,
                                     z_window: int = COINT_Z_WINDOW) -> pd.Series:
    """α·β 는 직전 ols_window 일(당일 제외)로만 추정 → 당일 스프레드 → 직전 z_window 일 기준 Z."""
    idx = close_a.index.intersection(close_b.index)
    la, lb = np.log(close_a.reindex(idx)), np.log(close_b.reindex(idx))
    mean_a = la.rolling(ols_window, min_periods=ols_window).mean()
    mean_b = lb.rolling(ols_window, min_periods=ols_window).mean()
    cov_ab = la.rolling(ols_window, min_periods=ols_window).cov(lb)
    var_b = lb.rolling(ols_window, min_periods=ols_window).var()
    beta = (cov_ab / var_b).shift(1)
    alpha = (mean_a - (cov_ab / var_b) * mean_b).shift(1)
    spread = la - (alpha + beta * lb)
    mu = spread.rolling(z_window, min_periods=z_window // 2).mean().shift(1)
    sd = spread.rolling(z_window, min_periods=z_window // 2).std().shift(1)
    return (spread - mu) / sd


def build_feature_frame(ticker_bars: dict[str, pd.DataFrame], target_bars: pd.DataFrame,
                        p: Params | None = None) -> pd.DataFrame:
    """종목별 일봉 → [date, ticker_id, 5 피처, close] 롱포맷 패널. 결측 행도 그대로 둔다
    (워밍업 구간 제거는 레이블 부착 후 한 번에)."""
    p = p or Params()
    frames = []
    for code, bars in ticker_bars.items():
        kf = kalman_features(bars, p)
        df = pd.DataFrame({
            "kalman_resid": compute_kalman_residual(kf),
            "price_z": compute_price_zscore(bars["close"]),
            "cusum": compute_cusum(kf, p),
            "hotelling_t2": compute_hotelling_t2(kf, p),
            "coint_z": compute_cointegration_residual_z(bars["close"], target_bars["close"]).reindex(bars.index),
            "close": bars["close"],
        }, index=bars.index)
        df.index.name = "date"
        df["ticker_id"] = code
        frames.append(df.reset_index())
    if not frames:
        return pd.DataFrame(columns=["date", "ticker_id", *FEATURE_COLS, "close"])
    out = pd.concat(frames, ignore_index=True)
    # 무한대(σ=0 구간 등)는 결측으로 통일
    out[FEATURE_COLS] = out[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return out[["date", "ticker_id", *FEATURE_COLS, "close"]].sort_values(["date", "ticker_id"],
                                                                          ignore_index=True)
