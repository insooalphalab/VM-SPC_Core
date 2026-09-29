"""텔레그램 첫 로그인(1회, 사용자가 직접 실행) — 휴대폰 번호 → 텔레그램 앱으로 온 인증 코드 → (설정했다면) 2단계 비밀번호.

  python reports\\tg_login.py

성공하면 state/telegram_reports.session 이 생기고, 이후 수집은 이 세션으로 자동 접속한다. 로그아웃하려면 그 파일을 지우거나
텔레그램 앱 설정 > 기기에서 해당 세션을 종료한다.
"""
from __future__ import annotations

import sys

from tg_client import CHANNELS, client


async def _check(c) -> None:
    me = await c.get_me()
    print(f"로그인 완료: {me.first_name or ''} (세션 저장됨)")
    for ch in CHANNELS:
        try:
            ent = await c.get_entity(ch)
            async for m in c.iter_messages(ent, limit=1):
                print(f"  {ch}: 접근 가능, 최신 글 #{m.id} ({m.date:%Y-%m-%d})")
        except Exception as e:
            print(f"  {ch}: 접근 실패 — {e}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    c = client()
    with c:                         # 처음이면 번호·코드·2단계 비밀번호를 콘솔에서 물어본다
        c.loop.run_until_complete(_check(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
