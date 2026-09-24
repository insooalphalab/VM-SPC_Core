"""V2 신호 계산: 4개 신호(수급 제외)의 연속값 + 연속 VM Score (설계문서 섹션 1·3).

V1(scorecard/signals.py)의 판정 로직(가격밴드·거래량·매물대·상대강도)을 그대로 재사용하되,
boolean pass/fail 대신 시그모이드로 부드럽게 이어지는 연속점수를 낸다. V1과 정확히 같은 경계값
(band_lo/hi, vol_z_min, profile_max_overhead_ratio, excess>0)에서 점수 0.5를 지나가도록 맞춰서,
"겉보기엔 연속점수이지만 실질은 V1과 같은 통과/미달 규칙"이라는 설계문서 3절의 성질을 유지한다.

V1과 다른 지점 하나: 데이터 부족(워밍업 전 등)일 때 V1은 "보수적으로 실패(0점)" 처리하지만,
V2는 명시적 결측(NaN)으로 남긴다. VM Score를 일별 리포트가 아니라 breadth 컨퓨전 매트릭스
백테스트에 쓰기 때문에, 워밍업 구간을 0점으로 깔면 초반 구간 통계가 왜곡된다.

모든 함수는 인과적이다(t 시점 값은 t까지의 데이터만 사용).
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
from v2_kalman import local_level, local_linear_trend

# Hotelling's T²(v2_compute_engine.build_t2_features)의 바스켓 평균 입력 신호.
# rs_excess(leave-one-out 초과수익)는 여기서 제외한다 — 바스켓 전체를 균등가중 평균하면
# Σ(r_i − mean_{j≠i} r_j) = 0 이 항등식으로 성립해(커버리지가 균일할 때) 그 열의 분산이
# 기계입실론 수준으로 사라지고, 공분산행렬이 특이(singular)해져 SVD가 수렴하지 못한다.
# (센서 수가 적고 결측이 거의 없는 바스켓에서 실제로 재현됨 — pinv 로도 못 막는 예외적 실패.)
# 개별 센서의 VM Score(rs_score)는 이 항등식과 무관하므로 영향받지 않는다.
T2_FEATURE_KEYS = ["band_z", "vol_z", "profile_ratio"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def soft_gate(margin: pd.Series, scale: float) -> pd.Series:
    """margin>0 이면 통과 쪽으로 매끈하게 이어지는 [0,1] 연속점수. margin이 NaN이면 결과도 NaN."""
    return pd.Series(_sigmoid(margin.to_numpy(dtype=float) / scale), index=margin.index)


def kalman_features(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    """일봉 → 가격밴드 z·기울기·σ, 거래량 z (V1 compute_features 의 가격/거래량 부분과 동일 계산)."""
    idx = bars.index
    nan = np.full(len(bars), np.nan)
    feat = pd.DataFrame({"close": bars["close"], "level": nan, "innov": nan, "band_z": nan,
                        "slope": nan, "slope_z": nan, "sigma": nan, "vol_z": nan}, index=idx)
    if len(bars) < p.kf_warmup + 5:
        return feat
    logc = np.log(bars["close"].to_numpy(dtype=float))
    kf = local_linear_trend(logc, warmup=p.kf_warmup, window=p.sigma_window, r_frac=p.kf_r_frac,
                            q_level_frac=p.kf_q_level_frac, q_slope_frac=p.kf_q_slope_frac)
    vk = local_level(np.log1p(bars["volume"].to_numpy(dtype=float)), warmup=p.kf_warmup,
                     window=p.sigma_window, r_frac=p.kf_r_frac, q_level_frac=p.kf_q_level_frac)
    feat["level"] = kf.level                      # 필터링된(사후) 레벨 — 차트 오버레이용 평활선
    feat["innov"] = logc - kf.pred                 # 잔차(로그가격, 1스텝 예측 대비) — CUSUM 입력
    feat["sigma"] = kf.sigma
    feat["band_z"] = feat["innov"] / feat["sigma"]
    feat["slope"] = kf.slope
    feat["slope_z"] = kf.slope / kf.sigma
    feat["vol_z"] = vk.innov / vk.sigma
    return feat


def _mass_in_range(low, high, vol, w, a: float, b: float) -> float:
    """일봉 거래량을 [저가,고가]에 균등 분포시켰을 때 가격구간 [a,b]에 걸리는 (감가 적용) 물량."""
    width = high - low
    overlap = np.clip(np.minimum(high, b) - np.maximum(low, a), 0, None)
    safe = np.where(width > 0, width, 1.0)
    frac = np.where(width > 0, overlap / safe, ((low >= a) & (low <= b)).astype(float))
    return float((vol * w * frac).sum())


def profile_ratio_series(bars: pd.DataFrame, p: Params) -> pd.Series:
    """일자별 상단/(상단+하단) 매물 비중 (V1 signal_profile 을 시계열 전체에 적용)."""
    out = pd.Series(np.nan, index=bars.index)
    low_a, high_a, vol_a, close_a = (bars[c].to_numpy(dtype=float) for c in ("low", "high", "volume", "close"))
    for i in range(len(bars)):
        if i + 1 < p.profile_min_bars:
            continue
        s = max(0, i + 1 - p.profile_lookback)
        low, high, vol, close = low_a[s:i + 1], high_a[s:i + 1], vol_a[s:i + 1], close_a[i]
        n = len(low)
        age = np.arange(n - 1, -1, -1, dtype=float)
        w = 0.5 ** (age / p.profile_decay_halflife) if p.profile_decay_halflife > 0 else np.ones(n)
        up = _mass_in_range(low, high, vol, w, close, close * (1 + p.profile_band_pct))
        down = _mass_in_range(low, high, vol, w, close * (1 - p.profile_band_pct), close)
        if up + down > 0:
            out.iloc[i] = up / (up + down)
    return out


def relative_strength_matrix(closes: dict[str, pd.Series], sensors: list[str], p: Params) -> pd.DataFrame:
    """센서별 N일 수익률 vs '자기 제외' 동일가중 그룹평균 초과수익 (V1 signal_relative_strength 벡터화).

    leave-one-out 평균을 반복문 없이 구하려고 (합계-자기)/(개수-1) 를 쓴다.
    """
    cols = [c for c in sensors if c in closes]
    rets = pd.DataFrame({c: closes[c].pct_change(p.rs_lookback) for c in cols})
    valid = rets.notna()
    n_valid = valid.sum(axis=1)
    total = rets.fillna(0.0).sum(axis=1)
    excess = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for c in rets.columns:
        peers_n = n_valid - valid[c].astype(int)
        peers_sum = total - rets[c].fillna(0.0)
        bench = peers_sum / peers_n.replace(0, np.nan)
        e = rets[c] - bench
        e[peers_n < p.rs_min_peers] = np.nan
        excess[c] = e
    return excess


def vm_scores(sensor_features: dict[str, pd.DataFrame], rs_excess: pd.DataFrame, p: Params) -> dict[str, pd.DataFrame]:
    """센서별 {날짜: raw 4신호·gate 점수·vm_score} DataFrame.

    네 신호 중 하나라도 결측이면 vm_score 도 결측 처리한다(모듈 docstring 참고).
    """
    out = {}
    for code, feat in sensor_features.items():
        band_gate = soft_gate(feat["band_z"] - p.band_lo, p.band_gate_scale) * \
            soft_gate(p.band_hi - feat["band_z"], p.band_gate_scale)
        slope_gate = soft_gate(feat["slope_z"] - p.min_slope, p.slope_gate_scale)
        price_band_score = band_gate * slope_gate
        volume_score = soft_gate(feat["vol_z"] - p.vol_z_min, p.vol_gate_scale)
        profile_score = soft_gate(p.profile_max_overhead_ratio - feat["profile_ratio"], p.profile_gate_scale)
        rs_col = rs_excess[code] if code in rs_excess.columns else pd.Series(np.nan, index=feat.index)
        rs_score = soft_gate(rs_col.reindex(feat.index), p.rs_gate_scale)

        vm = (p.w_price_band * price_band_score + p.w_volume * volume_score +
              p.w_profile * profile_score + p.w_rel_strength * rs_score)
        vm[price_band_score.isna() | volume_score.isna() | profile_score.isna() | rs_score.isna()] = np.nan

        out[code] = pd.DataFrame({
            "band_z": feat["band_z"], "vol_z": feat["vol_z"], "profile_ratio": feat["profile_ratio"],
            "rs_excess": rs_col.reindex(feat.index),
            "price_band_score": price_band_score, "volume_score": volume_score,
            "profile_score": profile_score, "rs_score": rs_score,
            "vm_score": vm, "buy_candidate": vm >= p.vm_threshold,
        })
    return out
