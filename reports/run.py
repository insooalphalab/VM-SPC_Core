"""매일 실행: 리포트 요약 새 글 수집(웹 미리보기, 한두 페이지) → 파싱 → 섹터 리포트 흐름 페이지(참고 자료).

  python reports/run.py
Epic AI(ked_epic_ai)는 과거 이력 수집을 로그인(API) 뒤로 미뤄서 지금은 버틀러만. 수집이 실패해도 저장된 데이터로 페이지는 만든다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "reports", _ROOT / "dart_events"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import logging
import sys

log = logging.getLogger("reports.run")
DAILY_CHANNELS = ["butler_works"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import collect_web
    import parse
    import render
    rc = 0
    for ch in DAILY_CHANNELS:
        try:
            log.info("%s 새 글 %d개", ch, collect_web.collect(ch))
        except Exception as e:  # 텔레그램 접속 실패 — 저장분으로 계속
            log.warning("%s 수집 실패: %s", ch, e)
            rc = 1
    df = parse.build()
    log.info("리포트 %d건", len(df))
    log.info("페이지: %s", render.build())
    return rc


if __name__ == "__main__":
    sys.exit(main())
