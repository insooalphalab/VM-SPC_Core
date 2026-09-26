"""레이블링 — 상대순위(2026-09-26부터 기본값) + 절대방향(명세 1.3/2.3, 비교용 옵션).

상대순위(`mode="relative"`, 기본): 그날 바스켓 안에서 상대적으로 잘 나갔는지로 라벨링한다 —
검증 결과(2026-09-26 세션), 동일 5피처·동일 Walk-Forward로 17개 바스켓을 절대방향/상대순위 양쪽
으로 돌려보니 예측력(edge) 자체가 더 크다는 근거는 없었다(Wilcoxon p=0.96, 바스켓 페어 비교).
대신 표본이 훨씬 더 많이 남아(데드존이 시장 전체가 조용한 날을 통째로 날리는 것과 달리, 상대
순위는 그날 바스켓 안에 상/하위가 항상 존재) 신뢰구간이 4배 가까이 좁아졌고(평균 CI폭 32%p→8%p),
95% CI 하한이 50%(동전던지기)를 넘는 바스켓이 1/17 → 16/17로 늘었다. 즉 "더 잘 맞힌다"가 아니라
"같은 크기의 약한 신호를 훨씬 적은 노이즈로 검출한다"는 개선이다 — 그래서 vm_spc/gate_decision.py
의 Gate 1 임계값도 그 실측 수준(0.60)으로 낮췄고, 통과해도 "고신뢰"가 아니라 "약한 방향성 참고
신호"로 취급한다. vm_spc/pipeline.py 의 `--label-mode absolute` 로 예전 방식과 비교해볼 수 있다.

절대방향(`mode="absolute"`, 비교용): R(t+1) >= +0.2% → 1, R(t+1) <= -0.2% → 0, 그 사이는
학습·검증에서 완전히 배제한다(슬리피지·거래비용·호가 스프레드 노이즈 구간). 경계값은
core.v2_config.Params.dead_zone 과 같다.
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

DEADZONE_UP = 0.002
DEADZONE_DOWN = -0.002


def label_target(next_day_return: float) -> int | None:
    if pd.isna(next_day_return):
        return None
    if next_day_return >= DEADZONE_UP:
        return 1
    if next_day_return <= DEADZONE_DOWN:
        return 0
    return None  # 학습/검증 제외


def next_day_returns(feature_frame: pd.DataFrame) -> pd.Series:
    """종목별 R(t+1) = close(t+1)/close(t) - 1. 마지막 날은 NaN(아직 모름)."""
    nxt = feature_frame.groupby("ticker_id")["close"].shift(-1)
    return nxt / feature_frame["close"] - 1


def attach_labels(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """R(t+1) 계산 후 label_target 적용, 데드존·미래미확정 행 drop."""
    df = feature_frame.copy()
    df["next_ret"] = next_day_returns(df)
    df["y"] = df["next_ret"].map(label_target)
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


# ── 상대순위 라벨링 (옵션) ──────────────────────────────────────────────
TOP_FRAC = 0.4     # 그날 바스켓 내 상위 40% 수익률 → 1(아웃퍼폼)
BOT_FRAC = 0.4     # 하위 40% → 0(언더퍼폼), 중간 20%는 상대적 동률 구간으로 제외
MIN_ACTIVE = 6     # 하루 유효종목 수가 이보다 적으면 순위 자체가 불안정해 그 날짜를 통째로 제외

LABEL_MODES = ("absolute", "relative")


def attach_relative_labels(feature_frame: pd.DataFrame, top_frac: float = TOP_FRAC,
                           bot_frac: float = BOT_FRAC, min_active: int = MIN_ACTIVE) -> pd.DataFrame:
    """날짜별로 R(t+1)의 바스켓 내 백분위 순위를 매겨, 상위 top_frac 은 1(아웃퍼폼),
    하위 bot_frac 은 0(언더퍼폼), 중간은 데드존과 같은 취지로 제외한다. 모듈 docstring의
    검증 결과 참고 — 예측력 자체가 절대방향보다 크다는 근거는 없고, 표본이 많이 남아
    같은 크기의 약한 신호를 훨씬 좁은 신뢰구간으로 검출한다는 게 실익이다."""
    df = feature_frame.copy()
    df["next_ret"] = next_day_returns(df)
    df = df.dropna(subset=["next_ret"]).copy()
    active = df.groupby("date")["ticker_id"].transform("count")
    df = df[active >= min_active].copy()
    pct_rank = df.groupby("date")["next_ret"].rank(pct=True, method="average")
    y = pd.Series(np.nan, index=df.index)
    y[pct_rank >= (1 - top_frac)] = 1.0
    y[pct_rank <= bot_frac] = 0.0
    df["y"] = y
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def build_labels(feature_frame: pd.DataFrame, mode: str = "relative", **kwargs) -> pd.DataFrame:
    """라벨링 모드 dispatch — vm_spc/pipeline.py 의 --label-mode 옵션에서 사용."""
    if mode == "absolute":
        return attach_labels(feature_frame)
    if mode == "relative":
        return attach_relative_labels(feature_frame, **kwargs)
    raise ValueError(f"알 수 없는 라벨링 모드: {mode!r} (가능한 값: {LABEL_MODES})")
