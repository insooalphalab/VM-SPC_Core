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

from tg_client import CHANNELS, SESSION, client
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


async def _collect(c) -> dict:
    out = {}
    for ch in CHANNELS:
        start, n = last_id(ch), 0
        with raw_path(ch).open("a", encoding="utf-8") as f:
            async for m in c.iter_messages(ch, min_id=start, reverse=True):
                if not m.message:
                    continue
                f.write(json.dumps({"id": m.id, "date": m.date.isoformat(), "text": m.message}, ensure_ascii=False) + "\n")
                n += 1
                if n % 2000 == 0:
                    log.info("%s %d개 (%s)", ch, n, m.date.date())
        out[ch] = n
        log.info("%s: 새 글 %d개", ch, n)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not _Path(str(SESSION) + ".session").exists():
        print("텔레그램 세션이 없습니다 — 먼저 python reports\\tg_login.py 로 로그인하세요.")
        return 2
    t0 = time.time()
    c = client()
    with c:
        res = c.loop.run_until_complete(_collect(c))
    log.info("완료 — %.1f분, %s", (time.time() - t0) / 60, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
