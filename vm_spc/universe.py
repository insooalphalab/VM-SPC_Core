"""Point-in-time 유니버스 (명세 1.5 / 2.4).

KRX 반도체 지수 정기변경 이력(시점별 구성종목)은 공개 API 로 제공되지 않고, 수기 이력 테이블도
아직 확보하지 못했다. 그래서 현재는 fallback 경로만 동작한다:
  POINT_IN_TIME_HISTORY 가 비어 있으면 → basket_watchlist.json 의 현재 센서 종목으로 전 기간을 근사
  (생존편향 경고를 로그와 결과 JSON 의 limitations 에 남긴다).
이력을 확보하면 POINT_IN_TIME_HISTORY 에 {"YYYY-MM-DD"(적용 개시일): [종목코드, ...]} 로 채우면 된다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import logging

log = logging.getLogger("vm_spc.universe")

# {적용 개시일: [종목코드...]} — 정기변경 이력. 비어 있으면 현재 구성종목 근사(fallback).
POINT_IN_TIME_HISTORY: dict[str, list[str]] = {}

SURVIVORSHIP_WARNING = ("Point-in-time 구성종목 이력 미확보 — 현재 바스켓 구성종목으로 전 기간을 근사했습니다. "
                        "기간 중 편출된 종목이 빠져 있어 성과가 낙관적으로 편향(생존편향)될 수 있습니다.")


def is_point_in_time() -> bool:
    return bool(POINT_IN_TIME_HISTORY)


def get_point_in_time_constituents(as_of_date: str, basket: dict) -> list[str]:
    if POINT_IN_TIME_HISTORY:
        eligible = [d for d in POINT_IN_TIME_HISTORY if d <= as_of_date]
        if eligible:
            return list(POINT_IN_TIME_HISTORY[max(eligible)])
    log.warning(SURVIVORSHIP_WARNING)
    return [s["code"] for s in basket["sensors"]]


def membership_mask(dates, tickers, basket: dict):
    """패널 각 행(date, ticker)이 그 날짜 기준 구성종목이었는지. fallback 이면 전부 True."""
    import numpy as np
    if not POINT_IN_TIME_HISTORY:
        return np.ones(len(dates), dtype=bool)
    cache: dict[str, set] = {}
    out = []
    for d, t in zip(dates, tickers):
        key = d.strftime("%Y-%m-%d")
        if key not in cache:
            cache[key] = set(get_point_in_time_constituents(key, basket))
        out.append(t in cache[key])
    return np.array(out, dtype=bool)
