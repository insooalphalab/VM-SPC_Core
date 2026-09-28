"""전 바스켓 고유 종목의 투자자 매매동향 수집(최초 5년 백필은 약 45분, 이후 증분은 종목당 1콜).

  python stock_track/collect_flow.py
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

from kis_client import RateLimitedCaller
from v2_config import load_baskets
from v2_data_collector import DEFAULT_HISTORY_DAYS

from stock_track.flow import update_flow

log = logging.getLogger("stock_track.collect_flow")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    codes = {}
    for b in load_baskets():
        for s in b["sensors"]:
            codes[s["code"]] = s["name"]
    caller, t0, failed = RateLimitedCaller(), time.time(), []
    for i, (code, name) in enumerate(codes.items(), 1):
        try:
            n, mode = update_flow(caller, code, DEFAULT_HISTORY_DAYS)
            log.info("[%d/%d] %s(%s): %s, 누적 %d일", i, len(codes), name, code, mode, n)
        except Exception as e:
            failed.append((code, str(e)[:80]))
            log.exception("%s(%s) 실패", name, code)
    log.info("완료 — %.1f분, 실패 %d건 %s", (time.time() - t0) / 60, len(failed), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
