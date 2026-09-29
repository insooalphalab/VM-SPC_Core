"""박스권 시나리오 규칙과 T² 판정(검증이력 9.18·9.27 사전 등록) — 검증(research/)과 화면(scenario/render_risk)이 함께 쓴다."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

BOX = 30             # 박스 = 직전 30거래일 고가·저가
FAIL_RECOVER = 3     # 가짜 이탈: 이탈 뒤 이 기간 안에 종가 복귀
RETEST_WITHIN = 10   # 돌파 리테스트: 돌파 뒤 이 기간 안에 박스 상단까지 되밀림
MAX_HOLD = 20        # 최대 보유 거래일
T2_WINDOW, T2_ALPHA = 60, 0.01     # 자체 T²: 칼만 band_z·slope_z·vol_z, 60일 창, 관리한계 α = 1%


def t2_flags(bars):
    """날짜별 "여러 지표가 동시에 평소와 다름"(자체 Hotelling T² > 관리한계). 그날까지의 데이터만 사용."""
    from v2_config import Params
    from v2_hotelling import hotelling_t2
    from v2_signals import kalman_features
    f = kalman_features(bars, Params())[["band_z", "slope_z", "vol_z"]]
    r = hotelling_t2(f, T2_WINDOW, T2_ALPHA)
    return (r.t2 > r.ucl).fillna(False).to_numpy()
