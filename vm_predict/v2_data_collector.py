"""Stage 1: 바스켓(센서+타겟 ETF) 일봉 수집 → data/{basket}/{code}.csv

  python v2_data_collector.py                              # basket_watchlist.json 의 첫 바스켓, 최근 3년
  python v2_data_collector.py --basket kospi_top10_to_etf
  python v2_data_collector.py --days 1500 --basket ...      # 수집 기간(달력일) 직접 지정
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
from datetime import datetime, timedelta

from kis_client import RateLimitedCaller, fetch_daily_bars, fetch_etf_listed_shares, fetch_nav_daily
from v2_config import KST, basket_codes, get_basket
from v2_datastore import append_shares, load_bars, load_nav, save_bars, save_nav

log = logging.getLogger("v2_data_collector")
DEFAULT_HISTORY_DAYS = 1825   # 약 5년(2026-09-28 3년→5년, 멀티 호라이즌 T+20 독립 표본 확보). 최초 백필에만 쓰이고,
                              # 기존 종목의 과거 구간 확장은 backfill_history.py 로 1회 수동 실행한다.
REFRESH_DAYS = 15             # 기존 데이터가 있어도 최근 이만큼은 다시 받는다(겹치는 구간으로 소급 조정 감지)
ADJUST_TOL = 0.005            # 겹치는 날짜 종가가 기존 값과 이 비율 이상 다르면 소급 조정으로 보고 전체 재수집


def _adjusted_since(existing, rows: list[dict]) -> bool:
    """액면분할·무상증자 등으로 KIS 수정주가가 과거 전체에 소급 조정됐는지 — 최근 구간만 덮어쓰면
    조정 전/후 가격이 섞여 시계열이 끊기므로, 겹치는 날짜의 종가를 비교해 감지한다."""
    for r in rows:
        ts = datetime.strptime(r["date"], "%Y%m%d")
        if ts in existing.index:
            old = existing.at[ts, "close"]
            if old > 0 and abs(r["close"] - old) / old > ADJUST_TOL:
                # 2026-09-28 정기 실행에서 67종목이 한꺼번에 감지돼 원인 추적용으로 남긴다(날짜·이전값·새값)
                log.info("소급 조정 감지: %s 종가 %s → %s (%+.2f%%)", r["date"], old, r["close"], (r["close"] / old - 1) * 100)
                return True
    return False


def update_bars(caller: RateLimitedCaller, basket_name: str, code: str, days: int) -> tuple[int, str] | None:
    """종목 하나를 최신화한다. 저장분이 있으면 최근 REFRESH_DAYS 만 받아 병합(증분), 없거나 소급
    조정이 감지되면 전체 재수집(백필). 반환: (받은 행 수, "증분"|"백필"|"조정 재수집") 또는 None."""
    today = datetime.now(KST).date()
    full_start = today - timedelta(days=days)
    existing = load_bars(basket_name, code)
    if existing is not None and not existing.empty:
        start = max(full_start, existing.index.max().date() - timedelta(days=REFRESH_DAYS))
        rows = fetch_daily_bars(caller, code, start, today)
        if not rows:
            return None
        if not _adjusted_since(existing, rows):
            save_bars(basket_name, code, rows, merge=True)
            return len(rows), "증분"
        mode = "조정 재수집"
    else:
        mode = "백필"
    rows = fetch_daily_bars(caller, code, full_start, today)
    if not rows:
        return None
    save_bars(basket_name, code, rows, merge=False)
    return len(rows), mode


def update_nav(caller: RateLimitedCaller, basket_name: str, code: str, days: int) -> tuple[int, str] | None:
    """타겟 ETF 의 NAV·괴리율 이력을 최신화하고, 오늘 상장좌수 스냅샷을 쌓는다.
    괴리율은 비율(%)이라 수정주가 소급 조정의 영향을 받지 않으므로 증분 병합만 한다."""
    today = datetime.now(KST).date()
    full_start = today - timedelta(days=days)
    existing = load_nav(basket_name, code)
    merge = existing is not None and not existing.empty
    start = max(full_start, existing.index.max().date() - timedelta(days=REFRESH_DAYS)) if merge else full_start
    rows = fetch_nav_daily(caller, code, start, today)
    if not rows:
        return None
    save_nav(basket_name, code, rows, merge=merge)
    # 상장좌수는 이력 API 가 없어 날짜를 마지막 NAV 거래일로 찍어 매일 한 줄씩 쌓는다(주말 재실행은 같은 날짜로 덮어씀)
    shares = fetch_etf_listed_shares(caller, code)
    if shares is not None:
        append_shares(basket_name, code, rows[-1]["date"], shares)
    return len(rows), "증분" if merge else "백필"


def collect(basket: dict, days: int) -> None:
    caller = RateLimitedCaller()
    for code, name in basket_codes(basket).items():
        res = update_bars(caller, basket["name"], code, days)
        if res is None:
            log.warning("%s(%s): 수집된 일봉 없음", name, code)
            continue
        log.info("%s(%s): (%s) %d개 수집", name, code, res[1], res[0])
    target = basket["target"]
    try:
        res = update_nav(caller, basket["name"], target["code"], days)
        if res is None:
            log.warning("%s(%s): NAV·괴리율 이력 없음", target["name"], target["code"])
        else:
            log.info("%s(%s): NAV·괴리율 (%s) %d개 수집", target["name"], target["code"], res[1], res[0])
    except Exception:
        # 괴리율은 보조 입력이라, 실패해도 기존 예측(일봉 기반)은 그대로 진행한다
        log.exception("%s(%s): NAV·괴리율 수집 실패 — 건너뜀", target["name"], target["code"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket")
    ap.add_argument("--days", type=int, default=DEFAULT_HISTORY_DAYS)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    basket = get_basket(args.basket)
    log.info("바스켓 '%s' 수집 시작 (타겟 %s + 센서 %d종목, %d일)",
             basket["name"], basket["target"]["name"], len(basket["sensors"]), args.days)
    collect(basket, args.days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
