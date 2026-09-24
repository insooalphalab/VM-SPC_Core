"""다변량 SPC — Hotelling's T² (설계문서 섹션 6).

날짜별 p차원 벡터(4개 신호의 바스켓 평균, 롤링 표준화)를 입력받아, 직전 W일(당일 미포함)로
추정한 평균·공분산 대비 T² 를 계산한다. 관리한계(UCL)는 표준 Hotelling T² 공식:
UCL = p(W-1)/(W-p) · F_α(p, W-p).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class T2Result:
    t2: pd.Series
    ucl: float
    contributions: pd.DataFrame   # 열=신호(p개), 행=날짜. 각 행의 합 = 그날의 t2 (정확한 분해, 근사 아님)


def _rolling_zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
    mu = x.rolling(window, min_periods=window).mean().shift(1)
    sd = x.rolling(window, min_periods=window).std().shift(1)
    return (x - mu) / sd


def hotelling_t2(features: pd.DataFrame, window: int, alpha: float, ridge: float = 1e-6) -> T2Result:
    """features: 열=신호(p개), 행=날짜. 특이행렬 방지로 numpy.linalg.pinv + 작은 능형(ridge) 보정 사용.

    표준화된 열은 대각성분이 1 근처라, ridge=1e-6은 정상적인 공분산행렬엔 사실상 영향이 없지만
    (거의) 상수인 열이 섞여 완전히 특이한 행렬이 됐을 때 SVD가 수렴하지 못하는 것을 막아준다.
    """
    p = features.shape[1]
    z = _rolling_zscore(features, window)
    zv = z.to_numpy()
    t2 = pd.Series(np.nan, index=features.index)
    contrib = pd.DataFrame(np.nan, index=features.index, columns=features.columns)
    for i in range(window, len(z)):
        win = zv[i - window:i]
        xt = zv[i]
        # 표준화(z-score) 분모가 0인 구간(어떤 신호가 그 60일 동안 완전히 평평했던 경우)은
        # Inf 를 만들 수 있어 NaN 과 함께 걸러낸다.
        if not (np.isfinite(xt).all() and np.isfinite(win).all()):
            continue
        cov = np.cov(win, rowvar=False) + ridge * np.eye(p)
        inv = np.linalg.pinv(cov)
        # 무조건부 기여도(unconditional contribution): x^T inv x = Σ_j x_j*(inv@x)_j 로 정확히
        # 분해된다 — 근사가 아니라 그날의 t2를 신호별로 남김없이 나눈 것(Σ contrib = t2).
        # 상관된 변수 간 기여를 완전히 분리하진 못하는 한계는 있지만(표준 FDC 실무에서도 쓰는
        # 근사 수준), "어떤 신호가 오늘 가장 벗어났는가"를 보는 1차 진단으로는 충분하다.
        terms = xt * (inv @ xt)
        t2.iloc[i] = float(terms.sum())
        contrib.iloc[i] = terms
    ucl = p * (window - 1) / (window - p) * stats.f.ppf(1 - alpha, p, window - p)
    return T2Result(t2=t2, ucl=float(ucl), contributions=contrib)
