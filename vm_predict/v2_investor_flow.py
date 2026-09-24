"""Stage 1.5(선택): 투자자별 순매수 수집 → data/{basket}/investor/{code}.csv, '기타기관' CUSUM 계산.

V2는 원래 일봉(FHKST03010100) 하나로만 데이터를 단순화했는데(v2_config.py 설계 결정), 개인/외국인/
기관합계 중 '기관합계'는 금융투자(증권사 자기매매·단기 성격)와 그 외 기관(연기금·투신·사모 등 —
방향성 포지션 성격이 더 강함)이 섞여 있어 신호가 희석될 수 있다는 가설을 확인하기 위해 별도로 추가.

  기타기관 순매수 = 기관합계 - 금융투자   (증권사 자기매매를 뺀 '진짜' 기관 수급)

이 시계열에 기존 v2_cusum과 동일한 방식(rolling_innovation으로 잔차·σ 추출 → CUSUM)을 그대로
적용해 검토할 수 있다 — breadth·칼만잔차와 같은 함수를 재사용(설계문서 "적용 대상 확장" 원칙).

  python v2_investor_flow.py collect --basket <바스켓명> [--days N]   # 수집만
  python v2_investor_flow.py cusum --basket <바스켓명> --code <종목코드>  # 수집된 데이터로 CUSUM 확인
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from kis_client import KISError, RateLimitedCaller
from v2_config import KST, Params, basket_codes, data_dir, get_basket
from v2_cusum import CusumResult, cusum, rolling_innovation
from v2_data_collector import DEFAULT_HISTORY_DAYS

log = logging.getLogger("v2_investor_flow")

INVESTOR_PATH = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
INVESTOR_TR = "FHPTJ04160001"                  # 종목별 투자자매매동향(일별), 1콜 최대 30건, 단위 백만원
FIELD_MAP = {
    "개인": "prsn_ntby_tr_pbmn",
    "외국인": "frgn_ntby_tr_pbmn",
    "기관합계": "orgn_ntby_tr_pbmn",
    "금융투자": "scrt_ntby_tr_pbmn",
}
FLOW_COLS = list(FIELD_MAP)


def fetch_investor_daily(caller: RateLimitedCaller, code: str, start: date, end: date,
                          max_pages: int = 60) -> pd.DataFrame:
    """[start, end] 투자자별 순매수(백만원)를 30건 한도에 맞춰 기준일을 과거로 옮기며 페이징 수집."""
    frames: list[pd.DataFrame] = []
    base = end
    for _ in range(max_pages):
        try:
            body = caller.get(INVESTOR_PATH, INVESTOR_TR, {
                "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
                "FID_INPUT_DATE_1": base.strftime("%Y%m%d"),
                "FID_ORG_ADJ_PRC": "", "FID_ETC_CLS_CODE": "",
            })
        except KISError as e:
            if "TIME LIMIT" in str(e) and base > start:
                base -= timedelta(days=1)   # 장중엔 '오늘' 기준 조회 불가 -> 하루 전 기준으로 재시도
                continue
            raise
        rows = [r for r in (body.get("output2") or []) if r.get("stck_bsop_date")]
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        oldest = min(r["stck_bsop_date"] for r in rows)
        if len(rows) < 30 or oldest <= start.strftime("%Y%m%d"):
            break
        base = datetime.strptime(oldest, "%Y%m%d").date() - timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["date", *FLOW_COLS])

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset="stck_bsop_date")
    df = pd.DataFrame({"date": raw["stck_bsop_date"]})
    for col, key in FIELD_MAP.items():
        df[col] = pd.to_numeric(raw.get(key), errors="coerce")
    df = df.dropna(subset=FLOW_COLS, how="all")
    dt = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df[(dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))]
    return df.sort_values("date").reset_index(drop=True)


def investor_flow_path(basket_name: str, code: str) -> Path:
    return data_dir() / basket_name / "investor" / f"{code}.csv"


def save_investor_flow(basket_name: str, code: str, df: pd.DataFrame) -> None:
    path = investor_flow_path(basket_name, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_investor_flow(basket_name: str, code: str) -> pd.DataFrame | None:
    path = investor_flow_path(basket_name, code)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"date": str})
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()[FLOW_COLS].astype(float)


def other_institutional_flow(flow: pd.DataFrame) -> pd.Series:
    """기타기관 순매수 = 기관합계 - 금융투자 (증권사 자기매매 제외한 나머지 기관 수급, 백만원)."""
    return (flow["기관합계"] - flow["금융투자"]).rename("기타기관")


def flow_cusum(flow_series: pd.Series, p: Params) -> CusumResult:
    """breadth·칼만잔차와 동일한 방식(rolling_innovation → cusum)을 수급 시계열에 재사용."""
    innov, sigma = rolling_innovation(flow_series, p.cusum_sigma_window)
    return cusum(innov, sigma, p.cusum_k, p.cusum_h)


def collect(basket: dict, days: int) -> None:
    caller = RateLimitedCaller()
    today = datetime.now(KST).date()
    start = today - timedelta(days=days)
    for code, name in basket_codes(basket).items():
        df = fetch_investor_daily(caller, code, start, today)
        if df.empty:
            log.warning("%s(%s): 수집된 수급 데이터 없음", name, code)
            continue
        save_investor_flow(basket["name"], code, df)
        log.info("%s(%s): %d개 수집", name, code, len(df))


def _cusum_report(basket_name: str, code: str, p: Params) -> None:
    flow = load_investor_flow(basket_name, code)
    if flow is None:
        print(f"{code}: 저장된 수급 데이터 없음 — 먼저 collect 실행 필요")
        return
    other = other_institutional_flow(flow)
    result = flow_cusum(other, p)
    n_up, n_down = int(result.alarm_up.sum()), int(result.alarm_down.sum())
    print(f"=== {code} 기타기관(기관합계-금융투자) CUSUM (k={p.cusum_k}, h={p.cusum_h}) ===")
    print(f"  기간: {other.index.min().date()} ~ {other.index.max().date()} ({len(other)}일)")
    print(f"  상방 알람(alarm_up): {n_up}건")
    print(f"  하방 알람(alarm_down): {n_down}건")
    if n_up:
        print(f"  최근 상방 알람일: {result.alarm_up[result.alarm_up].index.max().date()}")
    if n_down:
        print(f"  최근 하방 알람일: {result.alarm_down[result.alarm_down].index.max().date()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--basket")
    c.add_argument("--days", type=int, default=DEFAULT_HISTORY_DAYS)

    z = sub.add_parser("cusum")
    z.add_argument("--basket")
    z.add_argument("--code", required=True)

    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "collect":
        basket = get_basket(args.basket)
        log.info("바스켓 '%s' 수급 수집 시작 (타겟 %s + 센서 %d종목, %d일)",
                 basket["name"], basket["target"]["name"], len(basket["sensors"]), args.days)
        collect(basket, args.days)
    elif args.cmd == "cusum":
        basket = get_basket(args.basket)
        _cusum_report(basket["name"], args.code, Params())
    return 0


if __name__ == "__main__":
    sys.exit(main())
