"""시장 전체 프로그램매매 일별(차익·비차익) — KIS 프로그램매매 종합현황(일별, FHPPG04600001), 2016~.

  python stock_track/collect_market_program.py   → data/_market/program_kospi.csv
열: date, arbt_net(차익 순매수 대금), nabt_net(비차익 순매수 대금), whol_net(전체), 단위는 KIS 원값(백만 원).
코스피200 선물 베이시스는 만기 지난 월물 이력이 1년 정도라 10년 검증에 못 써서, 베이시스가 움직이는 차익 프로그램 금액을 직접 쓴다(9.50).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import sys
from datetime import datetime, timedelta

import pandas as pd

from kis_client import RateLimitedCaller, to_float
from v2_config import data_dir, last_complete_day

PATH = "/uapi/domestic-stock/v1/quotations/comp-program-trade-daily"
TR = "FHPPG04600001"


def out_path():
    d = data_dir() / "_market"
    d.mkdir(parents=True, exist_ok=True)
    return d / "program_kospi.csv"


def collect(start: str = "20160101") -> pd.DataFrame:
    c, rows = RateLimitedCaller(), {}
    cur = datetime.strptime(start, "%Y%m%d")
    end = last_complete_day()
    while cur.date() <= end:
        nxt = min(cur + timedelta(days=25), datetime.combine(end, datetime.min.time()))
        out = c.get(PATH, TR, {"FID_COND_MRKT_DIV_CODE": "J", "FID_MRKT_CLS_CODE": "K",
                               "FID_INPUT_DATE_1": cur.strftime("%Y%m%d"), "FID_INPUT_DATE_2": nxt.strftime("%Y%m%d")}).get("output") or []
        for x in out:
            d = x.get("stck_bsop_date")
            if d:
                rows[d] = {"date": d, "arbt_net": to_float(x.get("arbt_smtn_ntby_tr_pbmn")),
                           "nabt_net": to_float(x.get("nabt_smtn_ntby_tr_pbmn")), "whol_net": to_float(x.get("whol_smtn_ntby_tr_pbmn"))}
        cur = nxt + timedelta(days=1)
    df = pd.DataFrame([rows[k] for k in sorted(rows)])
    df.to_csv(out_path(), index=False, encoding="utf-8")
    return df


def load() -> pd.DataFrame:
    df = pd.read_csv(out_path(), dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").sort_index()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = collect()
    print(len(d), d["date"].min(), d["date"].max())
    print(d.tail(3).to_string(index=False))
