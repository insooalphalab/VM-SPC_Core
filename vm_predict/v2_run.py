"""전체 파이프라인 한 번에: Stage1(수집) → Stage2(연산) → Stage3(렌더링).

3단계를 따로 돌리는 게 귀찮을 때 이 파일 하나만 실행하면 된다. 각 Stage 파일(v2_data_collector.py
등)은 그대로 단독 실행도 가능 — 예를 들어 화면만 다시 손보고 싶으면 v2_render_dashboard.py만 돌리면
API를 재호출하지 않는다.

  python v2_run.py                              # basket_watchlist.json 의 모든 바스켓 전부
  python v2_run.py --basket semiconductor_to_etf
  python v2_run.py --no-fetch                   # 이미 수집된 data/*.csv 재사용 (API 호출 생략)
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

from v2_config import load_baskets
from v2_compute_engine import run_basket as compute_basket
from v2_data_collector import DEFAULT_HISTORY_DAYS, collect
from v2_render_dashboard import render_basket

log = logging.getLogger("v2_run")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket", help="생략 시 basket_watchlist.json 의 모든 바스켓을 순서대로 실행")
    ap.add_argument("--no-fetch", action="store_true", help="Stage1(일봉 수집) 생략, 저장된 data/*.csv 재사용")
    ap.add_argument("--days", type=int, default=DEFAULT_HISTORY_DAYS, help="수집 기간(달력일), 기본 약 3년")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    baskets = [b for b in load_baskets() if args.basket in (None, b["name"])]
    if not baskets:
        log.error("바스켓 없음: '%s' (basket_watchlist.json 확인)", args.basket)
        return 1

    failed = []
    for basket in baskets:
        log.info("=== 바스켓 '%s' (타겟 %s, 센서 %d종목) ===",
                 basket["name"], basket["target"]["name"], len(basket["sensors"]))
        if not args.no_fetch:
            collect(basket, args.days)
        if compute_basket(basket) != 0:
            failed.append(basket["name"])
            continue
        render_basket(basket)

    if failed:
        log.error("실패한 바스켓: %s", failed)
        return 1
    log.info("전체 완료 — results/{바스켓이름}/dashboard_v2.html 을 열어보세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
