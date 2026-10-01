"""텔레그램 여론 점수(easobi "텔레그램 심층 분석 Report") → 시장 점수·종목 점수 표.

글 형식(2026-09 확인):
  📝 텔레그램 심층 분석 Report (11시 07분) 기준
  🌡️ 센티먼트 낙관 (+40)
  💡 요약 ...
  🗣️ Top 키워드: ...
  🔥 주목받는 종목
  📈 S-Oil우 (+50)          ← 종목명 (점수), 다음 줄 이유
→ data/_reports/sentiment_market.csv (time, label, score, msg_id)
  data/_reports/sentiment_stock.csv  (time, name, code, score, msg_id) — 코드는 DART 상장사 이름 대조(해외 종목은 빠짐)
검증 전(검증이력 9.41)이라 화면에는 쓰지 않는다.

  python reports/sentiment.py
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

CHANNEL = "easobi"
KST = timezone(timedelta(hours=9))
MARKET = re.compile(r"센티먼트\s*([^(\n]*?)\s*\(([+-]?\d+(?:\.\d+)?)\)")
STOCK = re.compile(r"^[📈📉]\s*(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)\s*$", re.M)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    n2c = name_to_code()
    mk, st = [], []
    for m in load_raw(CHANNEL):
        t = m["text"]
        if "심층 분석 Report" not in t:
            continue
        when = datetime.fromisoformat(m["date"]).astimezone(KST).strftime("%Y-%m-%d %H:%M")
        mm = MARKET.search(t)
        if mm:
            mk.append({"time": when, "label": mm.group(1).strip(), "score": float(mm.group(2)), "msg_id": m["id"]})
        body = t.split("주목받는 종목", 1)[1] if "주목받는 종목" in t else ""
        for name, score in STOCK.findall(body):
            st.append({"time": when, "name": name.strip(), "code": n2c.get(name.replace(" ", "")),
                       "score": float(score), "msg_id": m["id"]})
    mk_df = pd.DataFrame(mk, columns=["time", "label", "score", "msg_id"]).sort_values("time")
    st_df = pd.DataFrame(st, columns=["time", "name", "code", "score", "msg_id"]).sort_values("time")
    out = data_dir() / "_reports"
    mk_df.to_csv(out / "sentiment_market.csv", index=False, encoding="utf-8")
    st_df.to_csv(out / "sentiment_stock.csv", index=False, encoding="utf-8")
    return mk_df, st_df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mk, st = build()
    print(f"시장 점수 {len(mk)}개 ({mk['time'].min()} ~ {mk['time'].max()}), 종목 점수 {len(st)}개 "
          f"(국내 코드 연결 {st['code'].notna().sum()}개, {st['code'].nunique()}종목)")
