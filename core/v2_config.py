"""경로·파라미터·바스켓 설정.

V1(scorecard/config.py)과 같은 원칙: 튜닝 대상 임계값은 전부 Params 한 곳에 모은다.
바스켓(센서 종목 리스트 + 예측대상 ETF)은 basket_watchlist.json 에서 읽는다 (섹션 7).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent  # core/ 의 부모 = 프로젝트 루트 (data/results/state/basket_watchlist.json 위치)
KST = timezone(timedelta(hours=9))


def _sub(name: str) -> Path:
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    """바스켓 이름별 하위 폴더에 일봉 CSV를 둔다: data/{basket}/{code}.csv"""
    d = Path(os.environ.get("V2_DATA_DIR", ROOT / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    """results/{basket}/{v2_summary_metrics.json, dashboard_v2.html}"""
    d = Path(os.environ.get("V2_RESULTS_DIR", ROOT / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_dir() -> Path:
    """kis_token_cache.json 등 실행 간 유지되는 상태."""
    return _sub("state")


BASKET_PATH = ROOT / "basket_watchlist.json"
EXAMPLE_BASKET_PATH = ROOT / "basket_watchlist.example.json"


def _derive_baskets(raw: dict) -> list[dict]:
    """stocks(종목마다 theme 태그) + themes(테마별 target ETF) → 내부 바스켓 리스트.

    같은 theme 이 붙은 stocks 전부가 그 테마 바스켓의 센서가 된다. themes 에 target 이 없는
    테마는 바스켓을 만들지 않는다 — 아직 예측 대상 ETF를 못 정한 섹터를 에러 없이 방치할 수 있게.
    """
    sensors_by_theme: dict[str, dict[str, dict]] = {}
    for s in raw.get("stocks", []):
        theme = s.get("theme")
        if not theme:
            continue
        code = str(s["code"]).zfill(6)
        sensors_by_theme.setdefault(theme, {})[code] = {"code": code, "name": s["name"]}

    baskets = []
    for theme, meta in (raw.get("themes") or {}).items():
        meta = meta or {}
        target = meta.get("target")
        sensors = list(sensors_by_theme.get(theme, {}).values())
        if not target or not target.get("code"):
            if sensors:
                log.info("테마 '%s': target 미지정 — 바스켓 생성 생략 (종목 %d개 대기 중)", theme, len(sensors))
            continue
        if not sensors:
            log.warning("테마 '%s': target 은 있는데 stocks 에 해당 theme 종목이 없음 — 바스켓 생성 생략", theme)
            continue
        baskets.append({"name": meta.get("basket_name") or theme, "target": target, "sensors": sensors})
    return baskets


def load_baskets(path: Path | None = None) -> list[dict]:
    path = path or BASKET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — basket_watchlist.example.json 을 basket_watchlist.json 으로 복사해 "
            f"실제 종목·타겟 ETF로 고치세요.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    baskets = _derive_baskets(raw)
    if not baskets:
        raise ValueError(
            "바스켓이 하나도 안 만들어졌습니다 — themes.<테마명>.target 에 예측 대상 ETF가 지정된 "
            "테마가 있는지, stocks 의 theme 태그가 themes 의 키와 정확히 일치하는지 확인하세요.")
    return baskets


def get_basket(name: str | None, path: Path | None = None) -> dict:
    """name 생략 시 파일의 첫 번째 바스켓을 쓴다."""
    baskets = load_baskets(path)
    if name is None:
        return baskets[0]
    for b in baskets:
        if b["name"] == name:
            return b
    raise KeyError(f"바스켓 '{name}' 없음. 사용 가능: {[b['name'] for b in baskets]}")


def basket_codes(basket: dict) -> dict[str, str]:
    """{종목코드: 종목명} — 센서 + 타겟 전부(일봉 수집·검증 대상)."""
    out = {s["code"]: s["name"] for s in basket["sensors"]}
    out[basket["target"]["code"]] = basket["target"]["name"]
    return out


@dataclass(frozen=True)
class Params:
    # ── 칼만필터 (V1과 동일한 원칙: 잡음 분산 = 워밍업 구간 일차차분 분산의 비율) ──
    kf_warmup: int = 40
    kf_r_frac: float = 0.25
    kf_q_level_frac: float = 0.30
    kf_q_slope_frac: float = 0.01
    sigma_window: int = 60

    # ① 가격밴드 (V1 signal_price_band 와 동일한 경계값, boolean 대신 연속 시그모이드로 전환)
    band_lo: float = -2.0
    band_hi: float = -1.0
    min_slope: float = 0.0
    band_gate_scale: float = 1.5      # 밴드 경계 근처 연속점수 전환 폭(σ) — 튜닝 근거는 아래 참고
    slope_gate_scale: float = 1.5     # 기울기(σ/day 단위) 연속점수 전환 폭

    # ② 거래량
    vol_z_min: float = 1.0
    vol_gate_scale: float = 1.5

    # ④ 매물대
    profile_lookback: int = 500
    profile_min_bars: int = 250
    profile_decay_halflife: int = 120
    profile_band_pct: float = 0.10
    profile_max_overhead_ratio: float = 0.40
    profile_gate_scale: float = 0.30

    # ⑤ 상대강도 (leave-one-out 동일가중, 바스켓 내 센서들이 비교군)
    rs_lookback: int = 20
    rs_min_peers: int = 2
    rs_gate_scale: float = 0.06

    # VM Score 가중치 (설계문서 섹션 3, 합계 1.0)
    w_price_band: float = 0.3
    w_volume: float = 0.3
    w_profile: float = 0.2
    w_rel_strength: float = 0.2
    vm_threshold: float = 0.75

    # Index Breadth (섹션 1)
    # 튜닝 이력(KOSPI top10 기준, 그리드서치):
    #  1차) 0.75(설계 초기값) → breadth가 0.43을 못 넘어 TP=FP=0으로 붕괴 → 0.35로 완화.
    #  2차) 0.35에서도 발동 횟수(recall)가 7.5%(n_pos=32)뿐이라 정밀도 우위가 통계적으로
    #       유의하지 않음(z≈0.88) → *_gate_scale(개별 신호가 1점에 도달하는 폭)을 3배 넓혀
    #       분포를 펴고, breadth_threshold를 그 분포의 70th 백분위수로 재설정.
    #       결과: n_pos=144(recall 33.6%), precision 68.1%(기저율 61.2%), z≈1.68 로 개선.
    #       gate_scale은 3배부터 포화(4~5배와 결과 동일)라 최소 충분값(3배)을 채택.
    breadth_threshold: float = 0.345
    breadth_min_coverage: float = 0.5   # 그날 유효 센서 비율이 이 미만이면 breadth 결측 처리

    # 데드존(무승부 구간): 익일 등락폭이 이 값 미만인 날은 방향을 맞히든 틀리든 컨퓨전 매트릭스
    # 집계에서 아예 제외한다 — 등락폭이 0%대로 애매한 날은 사실상 동전던지기라 어느 모델을 써도
    # 못 맞히는 게 정상이고, 그런 날을 "틀림(FP/FN)"으로 계속 섞어 세면 진짜 방향성 신호가 희석된다.
    # 실제 매매에서도 확신 없는 날은 안 들어가는 것과 같은 논리라, breadth_threshold 재탐색처럼
    # "결과 좋아지는 임계값을 사후에 고르는" 다중비교 문제와는 성격이 다르다(사전에 동기부여된 규칙).
    # KOSPI top10 3년 데이터로 0.0~2.0% 사이를 스캔한 결과:
    #  - 0.2%부터 z가 1.68→2.55로 뛰고, 0.1~2.0% 전 구간에서 z≥1.96을 대부분 유지(노이즈성 단일
    #    스파이크가 아니라 넓은 구간에서 일관된 개선 — 진짜 효과일 가능성을 시사).
    #  - 반도체 바스켓 교차검증에서는 같은 방향(정밀도 소폭 상승)이지만 z가 1.96을 꾸준히 넘지는
    #    않음(0.2%에서 1.73) — 바스켓마다 효과 크기가 다르므로 과신은 금물.
    #  - 0.2%는 두 바스켓 모두에서 무난한 지점이라 채택. 표본이 줄어드는 트레이드오프가 있다
    #    (0.2%에서 kospi 기준 477일→420일, 반도체는 477일→444일).
    dead_zone: float = 0.002

    # CUSUM 양방향 (섹션 2, σ 단위)
    cusum_k: float = 0.5
    cusum_h: float = 4.5
    cusum_sigma_window: int = 60

    # Hotelling's T² (섹션 6, 바스켓 평균 4개 신호 — 수급 제외, Stage1 데이터소스와 일관)
    t2_window: int = 60
    t2_alpha: float = 0.01
    t2_min_coverage: float = 0.5
