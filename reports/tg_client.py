"""텔레그램 사용자 계정 API 클라이언트(리포트 요약 채널 읽기 전용).

키: .streamlit/secrets.toml 의 TELEGRAM_API_ID / TELEGRAM_API_HASH. 세션 파일은 state/telegram_reports.session
(.gitignore 대상 — 계정 접근 권한이므로 절대 공유·커밋하지 않는다). 첫 로그인은 reports/tg_login.py 로 사용자가 직접.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from secrets_loader import get_secret

SESSION = _ROOT / "state" / "telegram_reports"          # Telethon 이 .session 을 붙인다
CHANNELS = ["butler_works", "ked_epic_ai", "aicorporateanalysisdeepdive", "shmstory"]   # 마지막은 시황(매크로·마감 요약)


def client():
    from telethon import TelegramClient
    SESSION.parent.mkdir(parents=True, exist_ok=True)
    return TelegramClient(str(SESSION), int(get_secret("TELEGRAM_API_ID")), get_secret("TELEGRAM_API_HASH"))
