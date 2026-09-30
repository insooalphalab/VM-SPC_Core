"""시황 한 줄 — 텔레그램 shmstory 채널의 가장 최근 시황 글 제목(마감 시황 또는 아침 "N월 N일 시황").

방향 판단에는 쓰지 않는 참고 정보(장세 한 줄 아래). 제목과 원문 링크만 남긴다 → results/reports/market_brief.json.

  python reports/market_brief.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import re
import sys
from datetime import datetime, timedelta, timezone

from parse import load_raw
from v2_config import results_dir

CHANNEL = "shmstory"
KST = timezone(timedelta(hours=9))
HEAD = re.compile(r"^(마감 시황|\d{1,2}월 \d{1,2}일 시황)\.\s*(.+)$")


def latest() -> dict | None:
    best = None
    for m in load_raw(CHANNEL):
        first = m["text"].strip().splitlines()[0].strip() if m["text"].strip() else ""
        h = HEAD.match(first)
        if h and (best is None or m["id"] > best["id"]):
            best = {"id": m["id"], "kind": "마감" if h.group(1) == "마감 시황" else "아침", "title": h.group(2).strip(),
                    "date": datetime.fromisoformat(m["date"]).astimezone(KST).strftime("%Y-%m-%d %H:%M"),
                    "link": f"https://t.me/{CHANNEL}/{m['id']}"}
    return best


def build() -> dict | None:
    b = latest()
    p = results_dir() / "reports" / "market_brief.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(b or {}, ensure_ascii=False), encoding="utf-8")
    return b


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(build())
