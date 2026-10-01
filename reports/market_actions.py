"""거래소 시장조치(텔레그램 darthacking) → data/_reports/market_actions.csv, 지금 걸려 있는 조치 조회.

DART OpenAPI로는 오지 않는 거래소 조치(투자경고·위험, 단기과열, 매매정지, 관리종목, 상장적격성 실질심사)를 종목별로 모은다.
방향 판단이 아니라 리스크 관리용 사실 정보 — 손절·가격 제한이 규칙대로 안 될 수 있어 오늘의 후보에서 빼고, 카드에 경고를 붙인다.

  level "제외" = 오늘의 후보에서 빼고 카드에 경고 · "경고" = 카드에 경고만(예고·조회공시)
  유효 기간 = 조치 종류별 달력일(아래 RULES). 같은 종목에 뒤에 나온 "해제" 글이 있으면 그 전 조치는 끝난 것으로 본다.

  python reports/market_actions.py
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

CHANNEL = "darthacking"
KST = timezone(timedelta(hours=9))
STATUS = re.compile(r"종목명\s*:\s*(.+?)\s*\n+\s*(\d{4}-\d{2}-\d{2})\s*:\s*(.+)")
DISC = re.compile(r"기업명:\s*(.+?)\(.*?\)\s*A?([0-9A-Z]{6})\s*\n\s*보고서명:\s*([^\n]+)")

# (찾을 말, 수준, 유효 달력일) — 위에서부터 먼저 맞는 것
RULES = [
    ("해제", "해제", 0),
    ("상장적격성", "제외", 90), ("실질심사", "제외", 90), ("관리종목", "제외", 90), ("상장폐지", "제외", 90),
    ("매매거래 정지", "제외", 10), ("매매거래정지", "제외", 10), ("거래정지", "제외", 10),
    ("투자위험", "제외", 20),
    ("지정예고", "경고", 10), ("예고", "경고", 10),
    ("투자경고종목지정", "제외", 20), ("투자경고종목 지정", "제외", 20), ("단기과열종목", "제외", 10),
    ("불성실공시", "경고", 30), ("조회공시요구", "경고", 5),
]
DISC_KEYS = ("기타시장안내", "조회공시요구", "관리종목", "상장적격성", "투자위험", "매매거래정지", "불성실공시")


def classify(text: str) -> tuple[str, int] | None:
    for key, level, days in RULES:
        if key in text:
            return level, days
    return None


def build() -> pd.DataFrame:
    n2c = name_to_code()
    rows = []
    for m in load_raw(CHANNEL):
        t = m["text"]
        posted = datetime.fromisoformat(m["date"]).astimezone(KST).date().isoformat()
        s = STATUS.search(t) if t.startswith("단기과열") else None
        if s:
            name, d, action = s.group(1).strip(), s.group(2), s.group(3).strip()
            code = n2c.get(name.replace(" ", ""))
        else:
            dm = DISC.search(t)
            if not dm or not any(k in t[:400] for k in DISC_KEYS):
                continue
            name, code, d = dm.group(1).strip(), dm.group(2), posted
            title = re.search(r"제목\s*:\s*([^\n]+)", t)
            action = (title.group(1).strip() if title else dm.group(3).strip())
            action = re.sub(r"(^|\s)br|br(\s|$)", " ", action)             # 원문 <br> 흔적
            action = re.sub(r"\s+", " ", action).strip()
        c = classify(action)
        if not code or not c:
            continue
        rows.append({"date": d, "code": code, "name": name, "action": action, "level": c[0], "days": c[1], "msg_id": m["id"]})
    df = pd.DataFrame(rows, columns=["date", "code", "name", "action", "level", "days", "msg_id"])
    df = df.drop_duplicates(["code", "date", "action"]).sort_values(["date", "msg_id"]).reset_index(drop=True)
    df.to_csv(data_dir() / "_reports" / "market_actions.csv", index=False, encoding="utf-8")
    return df


def active(asof: str | None = None) -> dict[str, dict]:
    """지금 걸려 있는 조치 {code: {date, action, level}} — 종목별 가장 최근 유효 조치(해제 뒤면 없음). 제외가 경고보다 우선."""
    p = data_dir() / "_reports" / "market_actions.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p, dtype={"code": str})
    today = pd.Timestamp(asof or datetime.now(KST).date())
    out = {}
    for code, g in d.groupby("code"):
        g = g.sort_values(["date", "msg_id"])
        last_clear = g.loc[g.level == "해제", "date"].max() if (g.level == "해제").any() else None
        live = g[(g.level != "해제") & (pd.to_datetime(g.date) + pd.to_timedelta(g.days, unit="D") >= today)]
        if last_clear is not None:
            live = live[live.date > last_clear]
        if live.empty:
            continue
        pick = live[live.level == "제외"].iloc[-1] if (live.level == "제외").any() else live.iloc[-1]
        out[code] = {"date": pick["date"], "action": pick["action"], "level": pick["level"]}
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = build()
    print(f"시장조치 {len(df)}건, {df['code'].nunique()}종목 · 수준별 {df['level'].value_counts().to_dict()}")
    a = active()
    print(f"지금 걸려 있음 {len(a)}종목:", {k: f"{v['level']} {v['date'][5:]} {v['action'][:24]}" for k, v in list(a.items())[:12]})
