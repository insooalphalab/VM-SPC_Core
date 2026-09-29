"""기관 세부 투자자별 순매수 수집 (검증이력 9.22) — KIS 종목별 투자자매매동향(일별), 1회 30거래일, 단위 백만원.

data/_investor_detail/{code}.csv — 종목 추적(stock_track/flow.py)·수급 검증·손절·수량 페이지가 함께 쓰는 유일한 수급 저장소.
  python stock_track/collect_investor_detail.py     # 센서 종목 191개, 2021-09~ (처음 약 50분)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import logging
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from kis_client import KISError, RateLimitedCaller, to_float
from v2_config import data_dir, last_complete_day, sensor_universe

log = logging.getLogger("collect_investor_detail")
PATH = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
TR = "FHPTJ04160001"
START = "20210901"
DETAIL = {
    "개인": "prsn", "외국인": "frgn", "기관합계": "orgn", "금융투자": "scrt", "투신": "ivtr", "사모": "pe_fund",
    "은행": "bank", "보험": "insu", "기타금융": "mrbn", "연기금": "fund", "기타법인": "etc_corp",
}


def detail_path(code: str):
    d = data_dir() / "_investor_detail"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{code}.csv"


def load_detail(code: str) -> pd.DataFrame | None:
    p = detail_path(code)
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()


def fetch(caller: RateLimitedCaller, code: str, until: str, stop_before: str) -> list[dict]:
    rows, cursor = {}, until
    for _ in range(80):
        try:
            out = caller.get(PATH, TR, {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code, "FID_INPUT_DATE_1": cursor,
                                        "FID_ORG_ADJ_PRC": "", "FID_ETC_CLS_CODE": ""}).get("output2") or []
        except KISError as e:
            if "TIME LIMIT" in str(e):          # 장중 '오늘' 기준 조회 불가 → 하루 전으로
                cursor = (datetime.strptime(cursor, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
                continue
            raise
        got = 0
        for x in out:
            d = x.get("stck_bsop_date")
            if not d or d in rows:
                continue
            rows[d] = {"date": d, **{k: to_float(x.get(f"{v}_ntby_tr_pbmn")) for k, v in DETAIL.items()}}
            got += 1
        if got == 0 or min(rows) <= stop_before:
            break
        cursor = (datetime.strptime(min(rows), "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    return [rows[k] for k in sorted(rows) if k >= stop_before]


def update(caller: RateLimitedCaller, code: str) -> int:
    until = last_complete_day().strftime("%Y%m%d")          # 장중 오늘 값은 미완성
    have = load_detail(code)
    if have is not None and not have.empty:
        last = have.index.max().strftime("%Y%m%d")
        new = fetch(caller, code, until, last) if last < until else []
    else:
        new = fetch(caller, code, until, START)
    if not new:
        return 0
    df = pd.DataFrame(new)
    if have is not None:
        df = pd.concat([have.reset_index().assign(date=lambda x: x["date"].dt.strftime("%Y%m%d")), df]).drop_duplicates("date", keep="last")
    df.sort_values("date").to_csv(detail_path(code), index=False, encoding="utf-8")
    return len(new)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    caller, t0, failed = RateLimitedCaller(), time.time(), []
    codes = sorted(sensor_universe())
    for n, code in enumerate(codes, 1):
        try:
            update(caller, code)
        except Exception as e:
            failed.append(code)
            log.warning("%s 실패: %s", code, e)
        if n % 20 == 0:
            log.info("%d/%d (%.1f분)", n, len(codes), (time.time() - t0) / 60)
    log.info("완료 — %.1f분, 실패 %s", (time.time() - t0) / 60, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
