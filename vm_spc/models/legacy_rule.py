"""Legacy Rule — 기존 스코어카드 판정 그대로 (명세 1.4).

풀링 패널은 종목 단위라, 바스켓 평균인 Breadth 대신 같은 식을 종목 하나에 적용한 값인
종목별 VM Score(core.v2_signals.vm_scores)에 동일 임계값 0.75(Params.vm_threshold)를 쓴다.
확률이 아니라 이진 판정이므로 S_base 와 S_core 가 같은 집합이 된다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import numpy as np

LEGACY_THRESHOLD = 0.75


def predict_legacy(breadth_score) -> np.ndarray:
    s = np.asarray(breadth_score, dtype=float)
    return np.where(np.nan_to_num(s, nan=-1.0) >= LEGACY_THRESHOLD, 1.0, 0.0)  # 확률 아님, 이진 판정
