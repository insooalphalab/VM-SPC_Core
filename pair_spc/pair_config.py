"""PAIR_SPC v2 경로·파라미터 (스펙: PAIR_SPC_v2_대표종목검증_스펙.md).

v2_config의 data_dir()/results_dir()를 그대로 재사용해 바스켓별 폴더 구조를 유지하되,
파일은 basket 하위의 pair/ 서브폴더에 둬서 기존 v2_summary_metrics.json 등과 섞이지 않게 한다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from dataclasses import dataclass
from pathlib import Path

from v2_config import data_dir, results_dir


def pair_results_dir(basket_name: str) -> Path:
    d = results_dir() / basket_name / "pair"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pair_data_dir(basket_name: str) -> Path:
    d = data_dir() / basket_name / "pair"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class PairParams:
    n_candidates: int = 5              # 섹터 ETF 상위 비중 후보 수 (스펙 2.3.1)
    history_days: int = 1095           # 약 3년 (V2 파이프라인과 동일 기준)

    # Stage 0 — 대표성 판정 (스펙 2.2, 잠정치. 섹션 6 "남겨둔 판단 사항" 참고)
    exself_corr_min: float = 0.7       # ex-self 동시상관 기준 — 스펙이 명시한 잠정치
    leadlag_days: tuple[int, ...] = (1, 2, 3)
    # 실측 결과 리드-래그(k=1~3) 상관은 대표성과 무관하게 거의 항상 0 근처(-0.1~0.06)로 나왔다 —
    # 일별 수익률 자체가 원래 자기상관이 거의 없는 성질이라, "리드-래그가 동시상관만큼 유지돼야
    # 한다"는 원래 기준은 구조적으로 항상 불합격하는 결함이 있었음(2026-09-24 확인 후 대표성
    # 판정에서 제외 — pair_representativeness.evaluate_candidate 참고). 리드-래그는 이제 참고
    # 정보로만 저장되고 판정에는 exself_corr만 쓰인다. leadlag_retain_ratio는 더 안 쓰이지만
    # 나중에 재도입할 경우를 위해 남겨둔다.
    leadlag_retain_ratio: float = 0.7

    # Stage 2 — 공적분/SPC (스펙 5절 Step4)
    adf_alpha: float = 0.05
    spc_window: int = 60
    spc_z_threshold: float = 2.0       # 관리도 이탈 판정 기준(σ)
    hysteresis_days: int = 3           # 연속 N거래일 확정돼야 상태전환(플리커링 방지, 스펙 5절 Step4)
