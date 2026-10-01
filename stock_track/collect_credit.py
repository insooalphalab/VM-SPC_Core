"""종목별 신용융자·대주 잔고 일별 수집 — KIS 국내주식 신용잔고 일별추이(FHPST04760000), 1회 30거래일.

  python stock_track/collect_credit.py            # 센서·관심 종목, 2021-09~
  python stock_track/collect_credit.py --validation [--since=20160101]   # 검증용 약 790종목(기본), 이력 확장 시 --since
  python stock_track/collect_credit.py --all      # 전 종목(보통주), 최근 2년~ — 쓰지 않음(사용자 결정: 검증용 종목만)
→ data/_credit/{code}.csv  열: date(매매일), loan_shares(융자 잔고 주수), loan_amt(융자 잔고 금액), loan_rate(잔고율 %),
  loan_gvrt(공여율 %), loan_new, loan_rdmp(신규·상환 주수), short_shares(대주 잔고 주수), short_rate(대주 잔고율 %).
신용 잔고는 결제일 기준이라 매매일보다 2~3일 늦게 확정된다(최근 며칠은 아직 없음).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import logging
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from kis_client import RateLimitedCaller, to_float
from v2_config import data_dir, last_complete_day, sensor_universe

log = logging.getLogger("collect_credit")
PATH = "/uapi/domestic-stock/v1/quotations/daily-credit-balance"
TR = "FHPST04760000"
START = "20210901"
START_ALL_YEARS = 2


def credit_path(code: str):
    d = data_dir() / "_credit"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{code}.csv"


def load_credit(code: str) -> pd.DataFrame | None:
    p = credit_path(code)
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()


def fetch(caller: RateLimitedCaller, code: str, until: str, stop_before: str) -> list[dict]:
    rows, cursor = {}, until
    while True:
        out = caller.get(PATH, TR, {"fid_cond_mrkt_div_code": "J", "fid_cond_scr_div_code": "20476",
                                    "fid_input_iscd": code, "fid_input_date_1": cursor}).get("output") or []
        got = 0
        for x in out:
            d = x.get("deal_date")
            if not d or d in rows:
                continue
            rows[d] = {"date": d, "loan_shares": to_float(x.get("whol_loan_rmnd_stcn")), "loan_amt": to_float(x.get("whol_loan_rmnd_amt")),
                       "loan_rate": to_float(x.get("whol_loan_rmnd_rate")), "loan_gvrt": to_float(x.get("whol_loan_gvrt")),
                       "loan_new": to_float(x.get("whol_loan_new_stcn")), "loan_rdmp": to_float(x.get("whol_loan_rdmp_stcn")),
                       "short_shares": to_float(x.get("whol_stln_rmnd_stcn")), "short_rate": to_float(x.get("whol_stln_rmnd_rate"))}
            got += 1
        if got == 0:
            break
        oldest = min(rows)
        if oldest <= stop_before:
            break
        cursor = (datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    return [rows[k] for k in sorted(rows) if k >= stop_before]


def update(caller: RateLimitedCaller, code: str, start: str = START) -> int:
    until = last_complete_day().strftime("%Y%m%d")
    have = load_credit(code)
    new = []
    if have is None or have.empty:
        new = fetch(caller, code, until, start)
    else:
        first, last = have.index.min().strftime("%Y%m%d"), have.index.max().strftime("%Y%m%d")
        if last < until:
            new += fetch(caller, code, until, last)
        if first > start:
            new += fetch(caller, code, (have.index.min() - timedelta(days=1)).strftime("%Y%m%d"), start)
    if not new:
        return 0
    df = pd.DataFrame(new)
    if have is not None:
        old = have.reset_index().assign(date=lambda x: x["date"].dt.strftime("%Y%m%d"))
        df = pd.concat([old, df]).drop_duplicates("date", keep="last")
    df.sort_values("date").to_csv(credit_path(code), index=False, encoding="utf-8")
    return len(new)


def targets(all_stocks: bool, validation: bool = False, since: str | None = None) -> tuple[list[str], str]:
    if validation:                                  # 검증용 약 790종목(전 종목은 하지 않음)
        from universe_all import validation_codes
        return validation_codes(), since or START
    if all_stocks:
        from universe_all import all_common_stocks
        start = (datetime.now() - timedelta(days=365 * START_ALL_YEARS)).strftime("%Y%m%d")
        return [c for c, _, _ in all_common_stocks()], start
    import json
    extra = []
    p = _ROOT / "scenario_targets.json"
    if p.exists():
        extra = [s["code"] for s in json.loads(p.read_text(encoding="utf-8")).get("stocks", [])]
    return sorted(set(sensor_universe()) | set(extra)), START


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    since = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--since=")), None)
    codes, start = targets("--all" in sys.argv, "--validation" in sys.argv, since)
    caller, t0, failed = RateLimitedCaller(), time.time(), []
    for n, code in enumerate(codes, 1):
        try:
            update(caller, code, start)
        except Exception as e:  # 한 종목 실패로 전체를 멈추지 않는다
            failed.append(code)
            log.warning("%s 실패: %s", code, e)
        if n % 50 == 0:
            log.info("%d/%d (%.1f분)", n, len(codes), (time.time() - t0) / 60)
    log.info("완료 — %d종목 %.1f분, 실패 %d개", len(codes), (time.time() - t0) / 60, len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
