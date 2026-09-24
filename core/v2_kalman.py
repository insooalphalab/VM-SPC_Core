"""칼만필터: 로컬 선형추세(레벨+기울기) / 로컬레벨. (V1 scorecard/kalman.py 와 동일 — 독립 구성을 위해 복사)

모두 인과적(causal)이다 — t 시점 출력은 t 이하 데이터만 사용한다. 잡음 분산도 워밍업 구간의
일차차분 분산으로만 정해서, 전체 시계열을 한 번 돌려도 각 날짜의 값이 그날 종가 확정 시점에
알 수 있던 정보와 같다(백테스트 look-ahead 방지).
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


@dataclass(frozen=True)
class KFResult:
    level: np.ndarray            # 필터링된 레벨 (사후, t까지 반영)
    slope: np.ndarray | None     # 필터링된 기울기 (로컬레벨 모델이면 None)
    pred: np.ndarray             # 1스텝 예측 레벨 (사전)
    innov: np.ndarray            # 잔차 = 관측 - 예측 레벨
    sigma: np.ndarray            # 잔차의 롤링 표준편차 (당일 제외, 워밍업 전은 NaN)


def _diff_var(y: np.ndarray, warmup: int) -> float:
    w = y[: max(warmup, 10) + 1]
    d = float(np.var(np.diff(w), ddof=1)) if len(w) > 2 else 0.0
    return max(d, 1e-12)


def _run(y: np.ndarray, F: np.ndarray, Q: np.ndarray, R: float, x0: np.ndarray, P0: np.ndarray):
    n, k = len(y), F.shape[0]
    level, slope = np.empty(n), np.empty(n)
    pred, innov = np.empty(n), np.empty(n)
    x, P = x0.copy(), P0.copy()
    for t in range(n):
        if t == 0:
            xp, Pp = x, P              # x0/P0 가 t=0 의 사전분포
        else:
            xp, Pp = F @ x, F @ P @ F.T + Q
        S = Pp[0, 0] + R
        v = y[t] - xp[0]
        K = Pp[:, 0] / S
        x = xp + K * v
        P = Pp - np.outer(K, Pp[0, :])
        P = (P + P.T) / 2
        pred[t], innov[t], level[t] = xp[0], v, x[0]
        slope[t] = x[1] if k > 1 else np.nan
    return level, slope, pred, innov


def _rolling_sigma(innov: np.ndarray, window: int, warmup: int) -> np.ndarray:
    s = pd.Series(innov).copy()
    s.iloc[:warmup] = np.nan                 # 초기 과도구간 잔차는 σ 에서 제외
    return s.rolling(window, min_periods=min(20, window)).std().shift(1).to_numpy()


def local_linear_trend(y, *, warmup: int, window: int, r_frac: float,
                       q_level_frac: float, q_slope_frac: float) -> KFResult:
    y = np.asarray(y, dtype=float)
    d = _diff_var(y, warmup)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_level_frac * d, q_slope_frac * d])
    R = r_frac * d
    level, slope, pred, innov = _run(y, F, Q, R, np.array([y[0], 0.0]), np.diag([R, d]))
    level[:warmup] = slope[:warmup] = np.nan   # 워밍업 구간은 값 신뢰 불가
    return KFResult(level, slope, pred, innov, _rolling_sigma(innov, window, warmup))


def local_level(y, *, warmup: int, window: int, r_frac: float, q_level_frac: float) -> KFResult:
    y = np.asarray(y, dtype=float)
    d = _diff_var(y, warmup)
    F = np.array([[1.0]])
    Q = np.array([[q_level_frac * d]])
    R = r_frac * d
    level, _, pred, innov = _run(y, F, Q, R, np.array([y[0]]), np.array([[R]]))
    level[:warmup] = np.nan
    return KFResult(level, None, pred, innov, _rolling_sigma(innov, window, warmup))
