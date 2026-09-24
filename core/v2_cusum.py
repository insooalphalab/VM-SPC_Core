"""CUSUM 누적합 관리도 (양방향, 설계문서 섹션 2).

k, h 는 σ 단위(무차원)로 받는다. 입력 innov 를 그때그때의 σ(rolling std, 당일 제외)로 나눠
표준화한 뒤 누적한다 — 개별종목 칼만 잔차·breadth 시계열 어디에나 같은 함수로 적용 가능
(설계문서 "적용 대상 확장" 원칙).
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
class CusumResult:
    z: pd.Series           # 표준화 잔차 (innov/sigma)
    c_pos: pd.Series
    c_neg: pd.Series
    alarm_up: pd.Series    # bool: 그날 C+ >= h
    alarm_down: pd.Series  # bool: 그날 C- <= -h


def cusum(innov: pd.Series, sigma: pd.Series, k: float, h: float) -> CusumResult:
    z = innov / sigma
    zv = z.to_numpy()
    c_pos, c_neg = np.zeros(len(zv)), np.zeros(len(zv))
    for i in range(len(zv)):
        prev_pos = c_pos[i - 1] if i > 0 else 0.0
        prev_neg = c_neg[i - 1] if i > 0 else 0.0
        if np.isnan(zv[i]):
            c_pos[i], c_neg[i] = 0.0, 0.0
            continue
        c_pos[i] = max(0.0, prev_pos + zv[i] - k)
        c_neg[i] = min(0.0, prev_neg + zv[i] + k)
    c_pos_s = pd.Series(c_pos, index=z.index)
    c_neg_s = pd.Series(c_neg, index=z.index)
    return CusumResult(z=z, c_pos=c_pos_s, c_neg=c_neg_s,
                       alarm_up=c_pos_s >= h, alarm_down=c_neg_s <= -h)


def rolling_innovation(x: pd.Series, window: int, min_periods: int = 20) -> tuple[pd.Series, pd.Series]:
    """관측치 자체의 '직전 window일 평균'을 예측치로 삼아 잔차·σ 를 만든다.

    칼만필터처럼 명시적 상태공간 모델이 없는 시계열(breadth 등)에 CUSUM을 적용하기 위한
    인과적(causal) 근사 — 당일이 자기 자신의 기준선 추정에 들어가지 않도록 shift(1) 한다.
    """
    pred = x.rolling(window, min_periods=min_periods).mean().shift(1)
    innov = x - pred
    sigma = innov.rolling(window, min_periods=min_periods).std().shift(1)
    return innov, sigma
