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

from kis_client import RateLimitedCaller, fetch_daily_bars
from v2_config import KST, basket_codes, get_basket
from v2_datastore import save_bars

log = logging.getLogger("v2_data_collector")
DEFAULT_HISTORY_DAYS = 1095   # 약 3년 — 워밍업(칼만·매물대 500봉) + 충분한 백테스트 구간 확보용


def collect(basket: dict, days: int) -> None:
    caller = RateLimitedCaller()
    today = datetime.now(KST).date()
    start = today - timedelta(days=days)
    for code, name in basket_codes(basket).items():
        rows = fetch_daily_bars(caller, code, start, today)
        if not rows:
            log.warning("%s(%s): 수집된 일봉 없음", name, code)
            continue
        save_bars(basket["name"], code, rows)
        log.info("%s(%s): %d개 수집", name, code, len(rows))


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
