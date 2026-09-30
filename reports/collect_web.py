"""공개 채널 웹 미리보기(t.me/s/채널)로 리포트 요약 원문 수집 — 로그인·인증 정보 불필요.

결과 형식은 reports/tg_collect.py(API)와 같다: data/_reports/raw/{채널}.jsonl (id, date(UTC ISO), text).
페이지(20개)마다 바로 저장하므로 중간에 끊겨도 받은 만큼 남고, 다음 실행이 이어서 받는다:
  1) 새 글: 최신 글부터 과거로, 이미 저장된 가장 큰 id 에 닿을 때까지
  2) 과거 채우기: 첫 글까지 다 받지 못했으면(state 파일) 저장된 가장 작은 id 아래부터 이어서
요청 사이 PAUSE 초를 쉰다(처음 전체 수집은 채널당 수십 분, 이후 증분은 한두 페이지).

  python reports/collect_web.py                   # 목표가가 있는 두 채널
  python reports/collect_web.py butler_works      # 채널 지정
  python reports/collect_web.py aicorporateanalysisdeepdive   # 뉴스 요약(최근 RECENT_DAYS일만)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import html
import json
import logging
import re
import sys
import time

import requests

from tg_collect import raw_path

log = logging.getLogger("reports.collect_web")
DEFAULT = ["butler_works", "ked_epic_ai"]
RECENT_DAYS = {"aicorporateanalysisdeepdive": 90}    # 뉴스 요약 채널: 글이 많아 최근 N일만(그 이전은 필요하면 API로)
PAUSE = 1.0
UA = {"User-Agent": "Mozilla/5.0 (VM-SPC Core personal research)"}


def to_text(fragment: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", fragment)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).replace("\xa0", " ").strip()


def fetch_page(ch: str, before: int | None) -> list[dict]:
    url = f"https://t.me/s/{ch}" + (f"?before={before}" if before else "")
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"{url} 실패")
    out = []
    for block in r.text.split('class="tgme_widget_message_wrap')[1:]:
        post = re.search(r'data-post="[^"/]+/(\d+)"', block)
        when = re.search(r'<time datetime="([^"]+)"', block)
        body = re.search(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', block, re.S)
        if post and when and body:
            out.append({"id": int(post.group(1)), "date": when.group(1), "text": to_text(body.group(1))})
    return out


def _saved_ids(ch: str) -> set[int]:
    p = raw_path(ch)
    if not p.exists():
        return set()
    with p.open(encoding="utf-8") as f:
        return {json.loads(x)["id"] for x in f}


def _state_path(ch: str):
    return raw_path(ch).with_suffix(".state.json")


def _walk(ch: str, before: int | None, stop_at: int, saved: set[int], f) -> tuple[int, bool]:
    """before 부터 과거로 내려가며 stop_at 이하에 닿거나 첫 글까지 저장. (새 글 수, 첫 글까지 닿았는지).
    RECENT_DAYS 채널은 그 기간 밖에 닿으면 '다 받음'으로 본다."""
    from datetime import datetime, timedelta, timezone
    since = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS[ch])).isoformat() if ch in RECENT_DAYS else ""
    n = 0
    while True:
        page = fetch_page(ch, before)
        if not page:
            return n, True
        for m in sorted(page, key=lambda x: x["id"]):
            if m["id"] > stop_at and m["id"] not in saved and m["date"] >= since:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
                saved.add(m["id"])
                n += 1
        f.flush()
        oldest = min(m["id"] for m in page)
        if since and min(m["date"] for m in page) < since:
            return n, True
        if oldest <= stop_at:
            return n, False
        if oldest <= 1:
            return n, True
        before = oldest
        if n and n % 1000 < 20:
            log.info("%s %d개 (%s)", ch, n, page[0]["date"][:10])
        time.sleep(PAUSE)


def collect(ch: str) -> int:
    saved = _saved_ids(ch)
    sp = _state_path(ch)
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"history_done": False}
    total = 0
    had_any = bool(saved)
    with raw_path(ch).open("a", encoding="utf-8") as f:
        n, reached_first = _walk(ch, None, max(saved) if saved else 0, saved, f)   # 1) 새 글 (처음이면 전체 이력)
        total += n
        if not had_any:
            state["history_done"] = reached_first
        if not state["history_done"] and saved:
            n, reached_first = _walk(ch, min(saved), 0, saved, f)    # 2) 과거 채우기
            total += n
            state["history_done"] = reached_first
    sp.write_text(json.dumps(state), encoding="utf-8")
    return total


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    for ch in (sys.argv[1:] or DEFAULT):
        n = collect(ch)
        log.info("%s: 새 글 %d개 (%.1f분)", ch, n, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
