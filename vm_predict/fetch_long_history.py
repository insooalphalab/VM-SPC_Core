"""섹터 ETF 장기 이력(상장일~) 수집 — 중장기 모멘텀 검증용(검증이력 9.16).

운영 도구들이 쓰는 data/{바스켓}/ (5년)은 건드리지 않고 data/_long_history/{코드}.csv 에 따로 저장한다.
대상: 전 바스켓의 고유 타겟 ETF + KODEX 200(기준 지수).

  python vm_predict/fetch_long_history.py            # 2005-01-01 부터
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import logging
import sys
import time
from datetime import date, datetime

from kis_client import RateLimitedCaller, fetch_daily_bars
from v2_config import KST, load_baskets
from v2_datastore import save_bars

log = logging.getLogger("fetch_long_history")
LONG_BASKET = "_long_history"
START = date(2005, 1, 1)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    codes = {"069500": "KODEX 200"}
    for b in load_baskets():
        codes.setdefault(b["target"]["code"], b["target"]["name"])
    caller, t0, failed = RateLimitedCaller(), time.time(), []
    today = datetime.now(KST).date()
    for code, name in codes.items():
        try:
            rows = fetch_daily_bars(caller, code, START, today)
            if rows:
                save_bars(LONG_BASKET, code, rows, merge=False)
                log.info("%s(%s): %d일 (%s~)", name, code, len(rows), rows[0]["date"])
            else:
                failed.append(code)
        except Exception as e:
            failed.append(code)
            log.exception("%s(%s) 실패: %s", name, code, e)
    log.info("완료 — %.1f분, 실패 %s", (time.time() - t0) / 60, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
