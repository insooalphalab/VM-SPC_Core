"""한투 Open API 클라이언트: 호출 간격 제한(RateLimitedCaller) + 일봉 조회.

V1(kis_client.py)에서 수급(investor flow) 조회를 뺀 버전 — V2는 Stage1 이 일봉(FHKST03010100)
하나로만 센서·타겟을 수집한다(설계문서 섹션 1 "데이터 파이프라인 단순화" 결정).

  python kis_client.py verify [바스켓이름]   # basket_watchlist.json 의 종목코드↔종목명 검증
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
from datetime import date, timedelta

import requests

from kis_auth import BASE_URL, auth_headers, get_token

log = logging.getLogger(__name__)

DAILY_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
DAILY_TR = "FHKST03010100"                     # 일봉: 1콜 최대 100건
STOCKINFO_PATH = "/uapi/domestic-stock/v1/quotations/search-stock-info"
STOCKINFO_TR = "CTPF1002R"                     # 상품기본조회(종목명·업종) — 종목코드 검증용
NAV_DAILY_PATH = "/uapi/etfetn/v1/quotations/nav-comparison-daily-trend"
NAV_DAILY_TR = "FHPST02440200"                 # ETF NAV 비교추이(일): 1콜 최대 100건, 과거 구간 지정 가능
ETF_PRICE_PATH = "/uapi/etfetn/v1/quotations/inquire-price"
ETF_PRICE_TR = "FHPST02400000"                 # ETF 현재가 — 상장좌수는 여기에만 있음(이력 API 없음)

DAILY_SPAN_DAYS = 140    # 100거래일 ≈ 140달력일 → 1콜당 100건 한도 안에 들어오도록 구간 분할

TOKEN_ERRORS = {"EGW00121", "EGW00123"}        # 유효하지 않은/만료된 토큰
RATE_ERRORS = {"EGW00201"}                     # 초당 거래건수 초과


class KISError(RuntimeError):
    pass


class RateLimitedCaller:
    """호출 사이 최소 간격을 보장하고, 속도제한/토큰만료 시 제한적으로 재시도한다."""

    def __init__(self, min_interval: float = 0.35, max_retries: int = 3):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def get(self, path: str, tr_id: str, params: dict) -> dict:
        refreshed = False
        for attempt in range(self.max_retries + 1):
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()

            resp = requests.get(BASE_URL + path, headers=auth_headers(tr_id), params=params, timeout=15)
            if resp.status_code == 429 and attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            body = resp.json() if resp.content else {}
            msg_cd = body.get("msg_cd", "")
            if msg_cd in TOKEN_ERRORS and not refreshed:
                get_token(force_refresh=True)   # 만료 토큰 1회 갱신 (발급 제한은 kis_auth 가 지킨다)
                refreshed = True
                continue
            if msg_cd in RATE_ERRORS and attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            if body.get("rt_cd") != "0":
                raise KISError(f"{tr_id} 실패: [{msg_cd}] {body.get('msg1')}")
            return body
        raise KISError(f"{tr_id} 재시도 초과")


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _to_float(s) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def fetch_daily_bars(caller: RateLimitedCaller, code: str, start: date, end: date) -> list[dict]:
    """[start, end] 일봉(수정주가)을 100건 한도에 맞춰 구간 분할 조회. 오름차순 반환."""
    rows: dict[str, dict] = {}
    cursor_end = end
    while cursor_end >= start:
        win_start = max(start, cursor_end - timedelta(days=DAILY_SPAN_DAYS))
        body = caller.get(DAILY_PATH, DAILY_TR, {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": _ymd(win_start),
            "FID_INPUT_DATE_2": _ymd(cursor_end),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        })
        got = 0
        for r in body.get("output2") or []:
            d = r.get("stck_bsop_date")
            o, h, l, c, v = (_to_float(r.get(k)) for k in
                             ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr", "acml_vol"))
            if not d or None in (o, h, l, c, v):
                continue
            rows[d] = {"date": d, "open": o, "high": h, "low": l, "close": c, "volume": v}
            got += 1
        if got == 0:      # 상장 이전 구간 등 → 더 거슬러 올라갈 필요 없음
            break
        cursor_end = win_start - timedelta(days=1)
    return [rows[k] for k in sorted(rows)]


def fetch_nav_daily(caller: RateLimitedCaller, code: str, start: date, end: date) -> list[dict]:
    """ETF [start, end] 일별 NAV·종가·괴리율(dprt, %). fetch_daily_bars 와 같은 구간 분할. 오름차순 반환."""
    rows: dict[str, dict] = {}
    cursor_end = end
    while cursor_end >= start:
        win_start = max(start, cursor_end - timedelta(days=DAILY_SPAN_DAYS))
        body = caller.get(NAV_DAILY_PATH, NAV_DAILY_TR, {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": _ymd(win_start),
            "FID_INPUT_DATE_2": _ymd(cursor_end),
        })
        got = 0
        for r in body.get("output") or []:
            d = r.get("stck_bsop_date")
            nav, close, dprt = (_to_float(r.get(k)) for k in ("nav", "stck_clpr", "dprt"))
            if not d or None in (nav, close, dprt) or nav <= 0:
                continue
            rows[d] = {"date": d, "nav": nav, "close": close, "dprt": dprt}
            got += 1
        if got == 0:
            break
        cursor_end = win_start - timedelta(days=1)
    return [rows[k] for k in sorted(rows)]


def fetch_etf_listed_shares(caller: RateLimitedCaller, code: str) -> float | None:
    """ETF 현재 상장좌수(설정·환매로 변함). 과거 이력 API 가 없어 매일 스냅샷으로 쌓는다."""
    body = caller.get(ETF_PRICE_PATH, ETF_PRICE_TR, {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": code})
    return _to_float((body.get("output") or {}).get("lstn_stcn"))


def fetch_stock_name(caller: RateLimitedCaller, code: str) -> str:
    """종목코드 → KIS 등록 종목약명 (없는 코드면 KISError)."""
    body = caller.get(STOCKINFO_PATH, STOCKINFO_TR, {"PRDT_TYPE_CD": "300", "PDNO": code})
    name = (body.get("output") or {}).get("prdt_abrv_name")
    if not name:
        raise KISError(f"{code}: 종목명 없음(존재하지 않는 코드?)")
    return name


def _norm_name(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _verify(basket_name: str | None) -> int:
    from v2_config import get_basket, basket_codes
    basket = get_basket(basket_name)
    targets = basket_codes(basket)
    caller, bad = RateLimitedCaller(), 0
    for code, name in targets.items():
        try:
            real = fetch_stock_name(caller, code)
        except Exception as e:
            print(f"[조회실패] {code} ({name}): {e}")
            bad += 1
            continue
        a, b = _norm_name(name), _norm_name(real)
        ok = a in b or b in a
        bad += not ok
        print(f"[{'OK' if ok else '불일치'}] {code}  목록={name}  KIS={real}")
    print()
    print(f"바스켓 '{basket['name']}' 총 {len(targets)}종목 중 문제 {bad}건")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) >= 2 and sys.argv[1] == "verify":
        sys.exit(_verify(sys.argv[2] if len(sys.argv) > 2 else None))
    else:
        print(__doc__)
