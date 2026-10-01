"""종목별 프로그램매매 일별 순매수 수집 (검증이력 9.21) — KIS 종목별 프로그램매매추이(일별), 1회 30거래일.

  python stock_track/collect_program.py        # 센서 종목 191개, 2021-09~ → data/_program/{code}.csv (처음 약 50분)
  python stock_track/collect_program.py --validation [--since=20160101]  # 검증용 약 790종목(기본), 이력 확장 시 --since
  python stock_track/collect_program.py --screen  # 오늘의 후보 모집단(코스피 시총 상위 200) 매일 증분 — 승률 모형의 이탈일 프로그램 z(9.79)
  python stock_track/collect_program.py --all  # 전 종목(보통주), 최근 2년~ — 쓰지 않음(사용자 결정: 검증용 종목만)
저장 열: date, prog_net(프로그램 순매수 대금, 원), value(거래대금, 원). 이미 받은 날짜는 건너뛰고 앞뒤만 채운다.
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

from kis_client import RateLimitedCaller, to_float
from v2_config import data_dir, last_complete_day, sensor_universe

log = logging.getLogger("collect_program")
PATH = "/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily"
TR = "FHPPG04650201"
START = "20210901"


def program_path(code: str):
    d = data_dir() / "_program"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{code}.csv"


def load_program(code: str) -> pd.DataFrame | None:
    p = program_path(code)
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()


def fetch(caller: RateLimitedCaller, code: str, until: str, stop_before: str) -> list[dict]:
    """until(포함)부터 과거로 30일씩, stop_before 이전에 닿으면 멈춤."""
    rows, cursor = {}, until
    while True:
        out = caller.get(PATH, TR, {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
                                    "FID_INPUT_DATE_1": cursor}).get("output") or []
        got = 0
        for x in out:
            d = x.get("stck_bsop_date")
            if not d or d in rows:
                continue
            rows[d] = {"date": d, "prog_net": to_float(x.get("whol_smtn_ntby_tr_pbmn")), "value": to_float(x.get("acml_tr_pbmn"))}
            got += 1
        if got == 0:
            break
        oldest = min(rows)
        if oldest <= stop_before:
            break
        cursor = (datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    return [rows[k] for k in sorted(rows) if k >= stop_before]


def update(caller: RateLimitedCaller, code: str, start: str = START) -> int:
    until = last_complete_day().strftime("%Y%m%d")          # 장중 오늘 값은 미완성
    have = load_program(code)
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
    df.sort_values("date").to_csv(program_path(code), index=False, encoding="utf-8")
    return len(new)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    caller, t0, failed = RateLimitedCaller(), time.time(), []
    since = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--since=")), None)
    if "--validation" in sys.argv:                    # 검증용 약 790종목(전 종목은 하지 않음 — 사용자 결정)
        _sys.path.insert(0, str(_ROOT / "stock_track"))
        from universe_all import validation_codes
        codes, start = validation_codes(), since or START
    elif "--screen" in sys.argv:                      # 오늘의 후보 모집단(코스피 시총 상위 200) — 매일 증분(승률 모형 9.79)
        _sys.path.insert(0, str(_ROOT / "scenario"))
        from screen import universe
        codes, start = [c for c, _ in universe()], START
    elif "--all" in sys.argv:                         # 코스피·코스닥 전 종목(보통주), 최근 2년~ — 쓰지 않음
        _sys.path.insert(0, str(_ROOT / "stock_track"))
        from universe_all import all_common_stocks
        codes = [c for c, _, _ in all_common_stocks()]
        start = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y%m%d")
    else:
        codes, start = sorted(sensor_universe()), START
    for n, code in enumerate(codes, 1):
        try:
            update(caller, code, start)
        except Exception as e:
            failed.append(code)
            log.warning("%s 실패: %s", code, e)
        if n % 20 == 0:
            log.info("%d/%d (%.1f분)", n, len(codes), (time.time() - t0) / 60)
    log.info("완료 — %.1f분, 실패 %s", (time.time() - t0) / 60, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
