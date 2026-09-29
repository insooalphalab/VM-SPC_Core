"""순매수 시계열의 CUSUM 경보 — 검증(research/validate_flow_cusum·joint_flow·flow_cusum_grid)과 화면(scenario/render_risk)이
같은 정의를 쓰도록 한 곳에 둔다(검증이력 9.21).

  x → 직전 window일 |x| 평균으로 나눠 단위 제거 → rolling_innovation(window) 잔차·σ → 양방향 CUSUM(k, h)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import pandas as pd

from v2_config import Params
from v2_cusum import CusumResult, cusum, rolling_innovation

WINDOW = 60
NORM_MIN = 40        # 정규화 평균의 최소 표본
INNOV_MIN = 20       # rolling_innovation 기본값(9.21 판정 기준)


def flow_cusum(x: pd.Series, window: int = WINDOW, h: float | None = None, k: float | None = None,
               norm_min: int = NORM_MIN, innov_min: int = INNOV_MIN) -> CusumResult:
    p = Params()
    z = x / x.abs().rolling(window, min_periods=norm_min).mean().shift(1)
    innov, sigma = rolling_innovation(z, window, min_periods=innov_min)
    return cusum(innov, sigma, p.cusum_k if k is None else k, p.cusum_h if h is None else h)


def onsets(res: CusumResult) -> tuple[pd.Series, pd.Series]:
    """경보가 켜진 첫날(상방, 하방)."""
    up, dn = res.alarm_up, res.alarm_down
    return up & ~up.shift(1, fill_value=False), dn & ~dn.shift(1, fill_value=False)
