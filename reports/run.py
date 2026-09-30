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
DAILY_CHANNELS = ["butler_works", "aicorporateanalysisdeepdive", "shmstory"]   # API 세션이 없을 때만 쓰는 웹 미리보기 대상


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import collect_web
    import market_brief
    import news
    import parse
    import render
    import tg_collect
    rc = 0
    api_ok = False
    if tg_collect.has_session():                # 로그인 세션이 있으면 API로 4개 채널 전부(빠짐없이)
        try:
            log.info("API 수집: %s", tg_collect.collect_all())
            api_ok = True
        except Exception as e:  # 세션 만료·접속 실패 — 웹 미리보기로 대신
            log.warning("API 수집 실패, 웹 미리보기로 대신: %s", e)
    if not api_ok:
        for ch in DAILY_CHANNELS:
            try:
                log.info("%s 새 글 %d개 (웹)", ch, collect_web.collect(ch))
            except Exception as e:  # 텔레그램 접속 실패 — 저장분으로 계속
                log.warning("%s 수집 실패: %s", ch, e)
                rc = 1
    df = parse.build()
    log.info("리포트 %d건", len(df))
    log.info("뉴스 종목 연결 %d건", len(news.build()))
    log.info("시황 한 줄: %s", market_brief.build())
    log.info("페이지: %s", render.build())
    return rc


if __name__ == "__main__":
    sys.exit(main())
