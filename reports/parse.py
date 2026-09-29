"""리포트 요약 원문(data/_reports/raw/*.jsonl) → 리포트 표 data/_reports/reports.csv.

열: date(리포트 작성일), code, name, broker, opinion, tp, dir(up/down/flat/new/none), price, source, msg_id
  - butler_works: 종목코드·작성일·증권사·의견·목표주가·(유지/상향/하향) 표시가 정형 — 주 데이터
  - ked_epic_ai : "목표주가 A원 >> B원 | 투자의견 X | 증권사" — 목표가 변경 리포트만, 종목명 → 코드는 DART 상장사 목록으로
같은 종목·증권사·작성일은 1건(버틀러 우선). 판정·집계는 reports/sector.py.

  python reports/parse.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "reports", _ROOT / "dart_events"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from kis_client import to_float
from v2_config import data_dir

KST = timezone(timedelta(hours=9))
BROKER_ALIAS = {"신한금융투자": "신한투자증권", "하나금융투자": "하나증권", "이베스트투자증권": "LS증권"}

B_HEAD = re.compile(r"^\s*(.+?)\s*\(([0-9A-Z]{6})\)", re.M)     # 2026년 신규 상장은 영문 섞인 코드(예: 0126Z0)
B_DATE = re.compile(r"작성일:\s*(\d{4})\.(\d{1,2})\.(\d{1,2})")
B_BROKER = re.compile(r"작성자:\s*([^\(\n]+?)\s*(?:\(|\n|$)")
B_OPINION = re.compile(r"투자의견:\s*([^\n]+)")
B_TP = re.compile(r"목표주가:\s*([\d,]+)\s*원\s*(?:\(([^)]*)\))?")
B_PRICE = re.compile(r"작성일 주가:\s*([\d,]+)\s*원")
K_HEAD = re.compile(r"목표주가 (상향|하향)\s*\|\s*(.+?)\s*\|")
K_LINE = re.compile(r"목표주가\s*([\d,]+)\s*원\s*>>\s*([\d,]+)\s*원\s*\|\s*투자의견\s*([^|]+?)\s*\|\s*([^\n]+)")
INITIATE = re.compile(r"커버리지 개시|신규 커버|커버 개시|Initiat|최초 커버", re.I)


def _dir_from_marker(marker: str | None, text: str) -> str:
    m = marker or ""
    if "상향" in m:
        return "up"
    if "하향" in m:
        return "down"
    if "신규" in m or "개시" in m or INITIATE.search(text):
        return "new"
    if "유지" in m:
        return "flat"
    return "none"


def _broker(s: str) -> str:
    s = s.strip()
    return BROKER_ALIAS.get(s, s)


def parse_butler(msg: dict) -> dict | None:
    t = msg["text"]
    head, date, broker = B_HEAD.search(t), B_DATE.search(t), B_BROKER.search(t)
    if not (head and broker):
        return None
    if date:
        d = f"{int(date.group(1)):04d}-{int(date.group(2)):02d}-{int(date.group(3)):02d}"
    else:                                   # "작성일: Invalid DateTime"(채널 쪽 오류) → 게시일로 대신
        d = datetime.fromisoformat(msg["date"]).astimezone(KST).date().isoformat()
    tp = B_TP.search(t)
    op = B_OPINION.search(t)
    pr = B_PRICE.search(t)
    return {"date": d, "code": head.group(2), "name": head.group(1).strip(), "broker": _broker(broker.group(1)),
            "opinion": op.group(1).strip() if op else None, "tp": to_float(tp.group(1)) if tp else None,
            "dir": _dir_from_marker(tp.group(2) if tp else None, t) if tp else "none",
            "price": to_float(pr.group(1)) if pr else None, "source": "butler_works", "msg_id": msg["id"]}


def parse_ked(msg: dict, name2code: dict[str, str]) -> dict | None:
    t = msg["text"]
    head, line = K_HEAD.search(t), K_LINE.search(t)
    if not (head and line):
        return None
    name = head.group(2).strip()
    code = name2code.get(name.replace(" ", ""))
    if not code:
        return None
    posted = datetime.fromisoformat(msg["date"]).astimezone(KST).date()
    return {"date": posted.isoformat(), "code": code, "name": name, "broker": _broker(line.group(4)),
            "opinion": line.group(3).strip(), "tp": to_float(line.group(2)),
            "dir": "up" if head.group(1) == "상향" else "down", "price": None, "source": "ked_epic_ai", "msg_id": msg["id"]}


def name_to_code() -> dict[str, str]:
    from client import corp_codes
    cc = corp_codes()
    return {str(n).replace(" ", ""): c for n, c in zip(cc["corp_name"], cc["stock_code"])}


def load_raw(ch: str) -> list[dict]:
    p = data_dir() / "_reports" / "raw" / f"{ch}.jsonl"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def build() -> pd.DataFrame:
    rows, stats = [], {}
    raw_b = load_raw("butler_works")
    pb = [r for r in (parse_butler(m) for m in raw_b) if r]
    stats["butler_works"] = (len(raw_b), len(pb))
    raw_k = load_raw("ked_epic_ai")
    n2c = name_to_code() if raw_k else {}
    pk = [r for r in (parse_ked(m, n2c) for m in raw_k) if r]
    stats["ked_epic_ai"] = (len(raw_k), len(pk))
    rows = pb + pk
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["src_rank"] = (df["source"] != "butler_works").astype(int)        # 같은 리포트면 버틀러 우선
    df = df.sort_values(["src_rank", "msg_id"]).drop_duplicates(["code", "broker", "date"], keep="first").drop(columns="src_rank")
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    out = data_dir() / "_reports" / "reports.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    df.attrs["stats"] = stats
    return df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = build()
    print({k: f"원문 {a}개 → 리포트 {b}건" for k, (a, b) in df.attrs.get("stats", {}).items()})
    if not df.empty:
        print(f"합계(중복 제거) {len(df)}건, {df['date'].min()} ~ {df['date'].max()}")
        print(df.groupby("source")["dir"].value_counts().unstack(fill_value=0))
