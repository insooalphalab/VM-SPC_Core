"""리포트 요약 채널 원문 수집 → data/_reports/raw/{채널}.jsonl (id, date(UTC ISO), text). 이미 받은 글 다음부터만 받는다.

  python reports/tg_collect.py           # 3개 채널 전체 이력(처음 몇 분), 이후 증분
파싱·섹터 집계는 따로(reports/parse.py). 원문을 남겨 두면 파서를 고쳐도 다시 받을 필요가 없다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import logging
import sys
import time

from tg_client import CHANNELS, RECENT_ONLY, SESSION, client
from v2_config import data_dir

log = logging.getLogger("reports.collect")


def raw_path(ch: str):
    d = data_dir() / "_reports" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ch}.jsonl"


def last_id(ch: str) -> int:
    p = raw_path(ch)
    if not p.exists():
        return 0
    last = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            last = max(last, json.loads(line)["id"])
    return last


def saved_ids(ch: str) -> set[int]:
    p = raw_path(ch)
    if not p.exists():
        return set()
    with p.open(encoding="utf-8") as f:
        return {json.loads(x)["id"] for x in f}


def _api_state(ch: str):
    return raw_path(ch).with_suffix(".api_state.json")


async def _collect(c, channels=None) -> dict:
    """1) 대조: 채널마다 한 번, 전체 글을 처음부터 끝까지 훑어 저장 안 된 글을 모두 채운다(웹 미리보기로 받다 빠진 글 포함).
    끝나면 api_state 에 reconciled 를 남긴다. 2) 그 뒤로는 저장된 가장 큰 id 다음 새 글만."""
    out = {}
    for ch in channels or CHANNELS:
        sp = _api_state(ch)
        state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        if not state.get("reconciled"):
            saved = saved_ids(ch)
            n = 0
            cutoff = None
            if ch in RECENT_ONLY:                   # 글이 매우 많은 채널은 최근 N일까지만 거슬러 올라간다
                from datetime import datetime, timedelta, timezone
                cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_ONLY[ch])
            with raw_path(ch).open("a", encoding="utf-8") as f:
                async for m in c.iter_messages(ch):
                    if cutoff and m.date < cutoff:
                        break
                    if m.message and m.id not in saved:
                        f.write(json.dumps({"id": m.id, "date": m.date.isoformat(), "text": m.message}, ensure_ascii=False) + "\n")
                        saved.add(m.id)
                        n += 1
            sp.write_text(json.dumps({"reconciled": True, "history_done": True}), encoding="utf-8")
            log.info("%s: 전체 대조로 빠진 글 %d개 채움 (전체 %d)", ch, n, len(saved))
            out[ch] = n
            continue
        saved = saved_ids(ch)
        n = 0
        with raw_path(ch).open("a", encoding="utf-8") as f:
            def write(m):
                nonlocal n
                if not m.message or m.id in saved:
                    return
                f.write(json.dumps({"id": m.id, "date": m.date.isoformat(), "text": m.message}, ensure_ascii=False) + "\n")
                saved.add(m.id)
                n += 1
                if n % 2000 == 0:
                    f.flush()
                    log.info("%s %d개 (%s)", ch, n, m.date.date())
            async for m in c.iter_messages(ch, min_id=max(saved, default=0), reverse=True):
                write(m)
        out[ch] = n
        log.info("%s: 새로 받은 글 %d개 (전체 %d)", ch, n, len(saved))
    return out


def has_session() -> bool:
    return _Path(str(SESSION) + ".session").exists()


def collect_all(channels=None) -> dict:
    """매일 실행(reports/run.py)에서 부르는 입구 — 세션으로 접속해 채널별 새 글(처음이면 전체 대조)."""
    c = client()
    with c:
        return c.loop.run_until_complete(_collect(c, channels))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not has_session():
        print("텔레그램 세션이 없습니다 — 먼저 python reports\\tg_login.py 로 로그인하세요.")
        return 2
    t0 = time.time()
    res = collect_all()
    log.info("완료 — %.1f분, %s", (time.time() - t0) / 60, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
