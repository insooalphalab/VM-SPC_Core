"""뉴스·리포트 요약 채널(aicorporateanalysisdeepdive + ked_epic_ai "주목해야할 뉴스") → 종목별 요약 표 data/_reports/news.csv.

글 형식(2026-09 확인):
  1줄 제목 · 2줄 "2026.09.30 | 매체"(뉴스) 또는 "2026.09.30 / 증권사 애널리스트"(리포트)
  "➤ 관련 기업" 다음 줄에 기업명(· 구분, 뉴스에만) · 맨 끝 해시태그
종목 연결(오탐을 줄이려고 본문 전체 검색은 하지 않음): 제목의 "(6자리 코드)" · 관련 기업 줄 · 해시태그 · 제목 첫 단어가
DART 상장사 이름과 정확히 같을 때. 방향 판정은 하지 않는다(검증 안 된 참고 정보).

  python reports/news.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "reports", _ROOT / "dart_events"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from parse import load_raw, name_to_code
from v2_config import data_dir

CHANNEL = "aicorporateanalysisdeepdive"
KST = timezone(timedelta(hours=9))
CODE_IN_TITLE = re.compile(r"\((\d{6})\)")
RELATED = re.compile(r"➤ 관련 기업\s*\n([^\n]+)")
TAG = re.compile(r"#([^\s#,]+)")
DATE_LINE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\s*([|/])\s*(.+)$")


def _key(s: str) -> str:
    return re.sub(r"\(.*?\)|\s", "", s)


def parse(msg: dict, n2c: dict[str, str], codes: set[str]) -> list[dict]:
    lines = [x.strip() for x in msg["text"].splitlines() if x.strip()]
    if len(lines) < 2:
        return []
    title = lines[0]
    dm = DATE_LINE.match(lines[1])
    if dm:
        d = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
        kind, source = ("뉴스" if dm.group(4) == "|" else "리포트"), dm.group(5).strip()
    else:
        d = datetime.fromisoformat(msg["date"]).astimezone(KST).date().isoformat()
        kind, source = "뉴스", ""
    found: set[str] = set(c for c in CODE_IN_TITLE.findall(title) if c in codes)
    names = []
    rm = RELATED.search(msg["text"])
    if rm:
        names += re.split(r"\s*[·,]\s*", rm.group(1))
    names += TAG.findall(msg["text"])
    names.append(title.split(" ")[0])
    for n in names:
        if _key(n) and _key(n) in _key(source):          # 리포트 글의 증권사 해시태그(#메리츠증권)는 종목이 아님
            continue
        c = n2c.get(_key(n))
        if c:
            found.add(c)
    return [{"date": d, "code": c, "kind": kind, "title": title, "source": source, "msg_id": msg["id"]} for c in sorted(found)]


KED_NEWS = "[epic AI 주목해야할 뉴스]"
KED_REL = re.compile(r"🏢 관련 기업\s*\n((?:\s*-\s*[^\n]+\n?)+)")


def parse_ked(msg: dict, n2c: dict[str, str]) -> list[dict]:
    """ked_epic_ai "주목해야할 뉴스": 머리 다음 줄 = 제목, "🏢 관련 기업" 아래 "- 기업명" 목록으로만 종목 연결."""
    t = msg["text"]
    if not t.startswith(KED_NEWS):
        return []
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    title = lines[1] if len(lines) > 1 else ""
    rm = KED_REL.search(t)
    if not rm or not title:
        return []
    codes = {n2c.get(_key(x.strip(" -"))) for x in rm.group(1).splitlines()} - {None}
    d = datetime.fromisoformat(msg["date"]).astimezone(KST).date().isoformat()
    return [{"date": d, "code": c, "kind": "뉴스", "title": title, "source": "ked", "msg_id": msg["id"], "channel": "ked_epic_ai"}
            for c in sorted(codes)]


def build() -> pd.DataFrame:
    raw = load_raw(CHANNEL)
    n2c = name_to_code()
    rows = [dict(r, channel=CHANNEL) for m in raw for r in parse(m, n2c, set(n2c.values()))]
    rows += [r for m in load_raw("ked_epic_ai") for r in parse_ked(m, n2c)]
    df = pd.DataFrame(rows, columns=["date", "code", "kind", "title", "source", "msg_id", "channel"])
    df = df.drop_duplicates(["channel", "code", "msg_id"]).drop_duplicates(["code", "date", "title"])
    df = df.sort_values(["date", "msg_id"]).reset_index(drop=True)
    df.to_csv(data_dir() / "_reports" / "news.csv", index=False, encoding="utf-8")
    df.attrs["n_raw"] = len(raw)
    return df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = build()
    print(f"원문 {df.attrs['n_raw']}개 → 종목 연결 {len(df)}건, {df['code'].nunique()}종목, {df['date'].min()} ~ {df['date'].max()}")
    print(df["kind"].value_counts().to_dict())
