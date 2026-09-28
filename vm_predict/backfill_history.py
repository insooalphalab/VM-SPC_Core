"""저장된 일봉·NAV 이력을 과거 방향으로 늘리는 1회성 수동 스크립트.

일일 증분 수집(update_bars)은 저장된 마지막 날짜 이후만 받으므로, 수집 기간을 늘려도 이미 데이터가 있는
종목의 과거 구간은 채워지지 않는다. 이 스크립트가 "저장된 가장 오래된 날짜 이전"을 한 번 받아 병합한다.
(2026-09-28, 멀티 호라이즌 T+20 검증에 필요한 독립 표본 확보를 위해 3년 → 5년)

- 과거 구간을 받을 때 기존 데이터와 OVERLAP_DAYS 만큼 겹치게 받아, 겹치는 날 종가가 다르면(수정주가 소급
  조정) 그 종목은 전체를 새로 받는다 — 조정 전/후 가격이 섞이지 않게.
- 상장이 목표 시작일보다 늦은 종목은 더 받을 게 없어 건너뛴다(1~2콜).
- 16:30 정기 실행과 겹치지 않을 때 수동으로 돌린다(KIS 호출 간격·토큰 공유).

  python vm_predict/backfill_history.py                 # 전 바스켓, 5년(DEFAULT_HISTORY_DAYS)
  python vm_predict/backfill_history.py --basket X --days 1825
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

from kis_client import RateLimitedCaller, fetch_daily_bars, fetch_nav_daily
from v2_config import KST, basket_codes, load_baskets
from v2_data_collector import DEFAULT_HISTORY_DAYS, _adjusted_since
from v2_datastore import load_bars, load_nav, save_bars, save_nav

log = logging.getLogger("backfill_history")
OVERLAP_DAYS = 20
SLACK_DAYS = 7   # 목표 시작일과 이 정도 차이면 이미 다 있는 것으로 본다(주말·휴일)


def extend_bars(caller: RateLimitedCaller, basket_name: str, code: str, days: int) -> str:
    today = datetime.now(KST).date()
    full_start = today - timedelta(days=days)
    existing = load_bars(basket_name, code)
    if existing is None or existing.empty:
        rows = fetch_daily_bars(caller, code, full_start, today)
        if not rows:
            return "데이터 없음"
        save_bars(basket_name, code, rows, merge=False)
        return f"신규 백필 {len(rows)}행"
    first = existing.index.min().date()
    if first <= full_start + timedelta(days=SLACK_DAYS):
        return "이미 충분"
    overlap_end = min(first + timedelta(days=OVERLAP_DAYS), existing.index.max().date())
    rows = fetch_daily_bars(caller, code, full_start, overlap_end)
    older = [r for r in rows if datetime.strptime(r["date"], "%Y%m%d").date() < first]
    if not older:
        return "상장 이후 전부 보유"
    if _adjusted_since(existing, rows):
        rows = fetch_daily_bars(caller, code, full_start, today)
        save_bars(basket_name, code, rows, merge=False)
        return f"소급 조정 감지 → 전체 재수집 {len(rows)}행"
    save_bars(basket_name, code, rows, merge=True)
    return f"과거 {len(older)}행 추가"


def extend_nav(caller: RateLimitedCaller, basket_name: str, code: str, days: int) -> str:
    today = datetime.now(KST).date()
    full_start = today - timedelta(days=days)
    existing = load_nav(basket_name, code)
    if existing is None or existing.empty:
        return "NAV 없음(일일 수집이 채움)"
    first = existing.index.min().date()
    if first <= full_start + timedelta(days=SLACK_DAYS):
        return "NAV 이미 충분"
    rows = fetch_nav_daily(caller, code, full_start, first - timedelta(days=1))
    if not rows:
        return "NAV 상장 이후 전부 보유"
    save_nav(basket_name, code, rows, merge=True)
    return f"NAV 과거 {len(rows)}행 추가"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket")
    ap.add_argument("--days", type=int, default=DEFAULT_HISTORY_DAYS)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    caller, t0, failed = RateLimitedCaller(), time.time(), []
    for basket in load_baskets():
        if args.basket not in (None, basket["name"]):
            continue
        for code, name in basket_codes(basket).items():
            try:
                log.info("[%s] %s(%s): %s", basket["name"], name, code, extend_bars(caller, basket["name"], code, args.days))
            except Exception as e:
                failed.append((basket["name"], code, str(e)[:80]))
                log.exception("[%s] %s(%s) 실패", basket["name"], name, code)
        t = basket["target"]
        try:
            log.info("[%s] %s NAV: %s", basket["name"], t["code"], extend_nav(caller, basket["name"], t["code"], args.days))
        except Exception as e:
            failed.append((basket["name"], t["code"] + " NAV", str(e)[:80]))
    log.info("완료 — %.1f분, 실패 %d건 %s", (time.time() - t0) / 60, len(failed), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
