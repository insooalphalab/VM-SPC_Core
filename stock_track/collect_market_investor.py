"""코스피 시장 전체 투자자별 일별 순매수 대금 — KIS 시장별 투자자매매동향(일별, FHPTJ04040000), 2015~.

  python stock_track/collect_market_investor.py   → data/_market/investor_kospi.csv
한 번 호출에 지정일부터 과거로 300거래일이 와서 뒤에서부터 거슬러 받는다. 응답의 투자자 항목을 모두 저장(사용자 요청):
금액(_amt, 백만 원)과 수량(_qty, 천 주) — 외국인(계·등록·비등록) · 개인 · 기관 계 · 금융투자 · 투신 · 사모 · 은행 · 보험 · 종금 · 연기금 ·
기타 금융 · 기타 법인 · 기타 계. 코스피 지수 시가·고가·저가·종가도 함께. 9.84 장세 군집 입력용.
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

PATH = "/uapi/domestic-stock/v1/quotations/inquire-investor-daily-by-market"
TR = "FHPTJ04040000"
START = "20150101"
WHO = {"frgn": "foreign", "frgn_reg": "foreign_reg", "frgn_nreg": "foreign_nreg", "prsn": "indiv", "orgn": "inst",
       "scrt": "fin_inv", "ivtr": "trust", "pe_fund": "pe", "bank": "bank", "insu": "insur", "mrbn": "merchant_bank",
       "fund": "pension", "etc_orgt": "etc_inst", "etc_corp": "etc_corp", "etc": "etc"}
PRICE = {"bstp_nmix_oprc": "kospi_open", "bstp_nmix_hgpr": "kospi_high", "bstp_nmix_lwpr": "kospi_low", "bstp_nmix_prpr": "kospi_close"}


def parse(x: dict) -> dict:
    """응답 한 줄 → 투자자별 금액·수량 전부. KIS 필드 이름이 항목마다 조금 달라(_tr_pbmn / _pbmn, _qty / _vol) 둘 다 찾는다."""
    row = {"date": x["stck_bsop_date"], **{v: to_float(x.get(k)) for k, v in PRICE.items()}}
    for k, v in WHO.items():
        row[f"{v}_amt"] = to_float(x.get(f"{k}_ntby_tr_pbmn", x.get(f"{k}_ntby_pbmn")))
        row[f"{v}_qty"] = to_float(x.get(f"{k}_ntby_qty", x.get(f"{k}_ntby_vol")))
    return row


def out_path():
    d = data_dir() / "_market"
    d.mkdir(parents=True, exist_ok=True)
    return d / "investor_kospi.csv"


def collect(start: str = START) -> pd.DataFrame:
    c, rows = RateLimitedCaller(), {}
    cur = last_complete_day().strftime("%Y%m%d")
    while cur >= start:
        out = c.get(PATH, TR, {"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": "0001", "FID_INPUT_DATE_1": cur,
                               "FID_INPUT_ISCD_1": "KSP", "FID_INPUT_DATE_2": cur, "FID_INPUT_ISCD_2": "0001"}).get("output") or []
        got = [x for x in out if x.get("stck_bsop_date")]
        if not got:
            break
        for x in got:
            rows[x["stck_bsop_date"]] = parse(x)
        oldest = min(x["stck_bsop_date"] for x in got)
        cur = (datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    df = pd.DataFrame([rows[k] for k in sorted(rows) if k >= start])
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
