"""DART 재무·공시 + 검증용 장기 가격 수집 (검증이력 9.17).

  python dart_events/collect.py              # 재무·공시·상장주식수 (몇 분)
  python dart_events/collect.py --prices     # + 종목 2015~ 일봉을 data/_long_history/ 에 (처음 한 번, 30분 안팎)
  python dart_events/collect.py --recent     # 매일: 최근 2년 재무·공시만 받아 기존 파일에 합침 + 밸류에이션 갱신
  python dart_events/collect.py --validation # 검증용 791종목 분기 재무(2015~)만 기존 financials.csv 에 합침 (공시·주식수는 안 건드림)

저장: data/_dart/financials.csv, events.csv, shares.csv. 운영 5년 데이터(data/{바스켓}/)는 건드리지 않는다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "dart_events"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import logging
import sys
import time
from datetime import date, datetime

import pandas as pd

from client import corp_codes, dart_dir, get
from kis_client import RateLimitedCaller, fetch_daily_bars, to_float
from v2_config import KST, LONG_HISTORY, last_complete_day, sensor_universe
from v2_datastore import load_bars, save_bars

log = logging.getLogger("dart_collect")
START_YEAR = 2015
REPORTS = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}      # 1분기·반기·3분기·사업보고서 → 분기 번호
ACCOUNTS = {"매출액": "rev", "영업이익": "op", "당기순이익": "ni", "당기순이익(손실)": "ni", "자본총계": "equity"}
EVENT_TYPES = {"자기주식취득": "buyback", "유상증자결정": "rights", "전환사채권발행결정": "cb"}
LONG_BASKET = LONG_HISTORY
PRICE_START = date(2015, 1, 1)
BENCH = ("069500", "229200")


def targets() -> dict[str, str]:
    """수집 대상 = 바스켓 센서 종목 + 박스권 페이지 목록(scenario_targets.json). 검증(validate.py)은 센서 종목만 쓴다."""
    out = sensor_universe()
    p = _ROOT / "scenario_targets.json"
    if p.exists():
        import json
        for x in json.loads(p.read_text(encoding="utf-8")).get("stocks", []):
            out.setdefault(str(x["code"] if isinstance(x, dict) else x).zfill(6), "")
    return out


def _save(new: pd.DataFrame, name: str, keys: list[str], merge: bool) -> pd.DataFrame:
    """merge=True 면 기존 파일과 합치고 같은 키는 새 값으로(재무 정정·재공시 반영), 아니면 통째로 덮어쓴다."""
    path = dart_dir() / name
    if merge and path.exists():
        old = pd.read_csv(path, dtype={"stock_code": str, "rcept_dt": str, "rcept_no": str})
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last")
    new.to_csv(path, index=False, encoding="utf-8")
    return new


def collect_financials(cc: pd.DataFrame, from_year: int = START_YEAR, merge: bool | None = None) -> pd.DataFrame:
    corps = cc["corp_code"].tolist()
    rows, this_year = [], datetime.now(KST).year
    for year in range(from_year, this_year + 1):
        for rc, q in REPORTS.items():
            for i in range(0, len(corps), 100):
                body = get("fnlttMultiAcnt", corp_code=",".join(corps[i:i + 100]), bsns_year=str(year), reprt_code=rc)
                for x in body.get("list", []):
                    acc = ACCOUNTS.get(x["account_nm"].replace(" ", ""))
                    if acc is None:
                        continue
                    rows.append({"stock_code": x["stock_code"], "year": year, "q": q, "rcept_dt": x["rcept_no"][:8],
                                 "fs_div": x["fs_div"], "account": acc,
                                 "amount": to_float(x.get("thstrm_amount")), "add_amount": to_float(x.get("thstrm_add_amount"))})
        log.info("재무 %d년 완료 (누적 %d행)", year, len(rows))
    keys = ["stock_code", "year", "q", "fs_div", "account"]
    df = pd.DataFrame(rows).drop_duplicates(keys, keep="first")
    return _save(df, "financials.csv", keys, merge=from_year > START_YEAR if merge is None else merge)


def collect_events(cc: pd.DataFrame, from_year: int = START_YEAR) -> pd.DataFrame:
    rows, today = [], datetime.now(KST).strftime("%Y%m%d")
    for n, (corp, code) in enumerate(zip(cc["corp_code"], cc["stock_code"]), 1):
        page = 1
        while True:
            body = get("list", corp_code=corp, bgn_de=f"{from_year}0101", end_de=today, pblntf_ty="B",
                       page_no=str(page), page_count="100")
            for x in body.get("list", []):
                name = x["report_nm"]
                typ = next((t for key, t in EVENT_TYPES.items() if key in name), None)
                if typ and "정정" not in name and "처분" not in name:
                    rows.append({"stock_code": code, "rcept_dt": x["rcept_dt"], "type": typ, "report_nm": name.strip(),
                                 "rcept_no": x["rcept_no"]})
            if page >= int(body.get("total_page") or 1):
                break
            page += 1
        if n % 50 == 0:
            log.info("공시 %d/%d 종목", n, len(cc))
    df = pd.DataFrame(rows, columns=["stock_code", "rcept_dt", "type", "report_nm", "rcept_no"])
    return _save(df, "events.csv", ["rcept_no"], merge=from_year > START_YEAR)


def collect_shares(caller: RateLimitedCaller, codes: list[str]) -> pd.DataFrame:
    """현재 상장주식수(바스켓 시가총액 근사용). KIS 주식현재가."""
    rows = []
    for code in codes:
        body = caller.get("/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100",
                          {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": code})
        v = to_float((body.get("output") or {}).get("lstn_stcn"))
        if v:
            rows.append({"stock_code": code, "shares": v})
    df = pd.DataFrame(rows)
    df.to_csv(dart_dir() / "shares.csv", index=False, encoding="utf-8")
    return df


def collect_prices(caller: RateLimitedCaller, codes: list[str]) -> list[str]:
    failed, today = [], last_complete_day()          # 장 마감 전이면 어제까지(미완성 오늘 봉 저장 방지)
    for n, code in enumerate(list(BENCH) + codes, 1):
        have = load_bars(LONG_BASKET, code)
        if have is not None and have.index[0].date() <= date(2015, 1, 10) and (today - have.index[-1].date()).days < 5:
            continue
        try:
            rows = fetch_daily_bars(caller, code, PRICE_START, today)
            if rows:
                save_bars(LONG_BASKET, code, rows, merge=False)
            else:
                failed.append(code)
        except Exception as e:  # 한 종목 실패로 전체를 멈추지 않는다
            failed.append(code)
            log.warning("%s 가격 실패: %s", code, e)
        if n % 20 == 0:
            log.info("가격 %d/%d", n, len(codes) + len(BENCH))
    return failed


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--recent", action="store_true")
    ap.add_argument("--validation", action="store_true")
    args = ap.parse_args()
    if args.validation:                                    # 검증 표본(9.62~)용 — 전 종목이 아니라 검증용 791종목만
        _sys.path.insert(0, str(_ROOT / "stock_track"))
        from universe_all import validation_codes
        t0, cc = time.time(), corp_codes()
        cc = cc[cc["stock_code"].isin(set(validation_codes()))]
        log.info("검증용 %d종목 분기 재무 수집", len(cc))
        fin = collect_financials(cc, START_YEAR, merge=True)
        log.info("완료 — %.1f분, financials.csv %d행 · %d종목", (time.time() - t0) / 60, len(fin), fin["stock_code"].nunique())
        return 0
    from_year = datetime.now(KST).year - 1 if args.recent else START_YEAR
    t0, uni = time.time(), targets()
    cc = corp_codes()
    cc = cc[cc["stock_code"].isin(uni)]
    log.info("대상 %d종목 중 DART 고유번호 %d개", len(uni), len(cc))
    fin = collect_financials(cc, from_year)
    ev = collect_events(cc, from_year)
    log.info("재무 %d행, 공시 사건 %d건 %s", len(fin), len(ev), ev["type"].value_counts().to_dict())
    caller = RateLimitedCaller()
    collect_shares(caller, sorted(uni))
    failed = collect_prices(caller, sorted(sensor_universe())) if args.prices else []
    if args.recent:
        from valuation import run_all
        log.info("밸류에이션 %d개 바스켓 갱신", len(run_all()))
    log.info("완료 — %.1f분, 가격 실패 %s", (time.time() - t0) / 60, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
