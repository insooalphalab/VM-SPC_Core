"""Stage 0: ETF 전체 목록에서 섹터 대표 ETF를 찾아 basket_watchlist.json 후보를 만든다.

설계 근거는 KIS_ETF_섹터스캐너_설계노트.md. 이 스캐너는 basket_watchlist.json 을 직접 덮어쓰지
않는다 — 결과를 basket_watchlist.scanned.json 에 별도로 써서, 사람이 검토 후 실제 파일에
수동으로 반영하는 걸 전제로 한다(설계노트 "남는 수동 작업" 항목: KIS 분류 라벨 변경·신상품
출시에 자동 필터가 흔들릴 수 있어, 최종 확정 리스트는 하드코딩하고 스캐너는 신규 후보 점검용으로만 쓴다).

  python v2_etf_scanner.py                         # 당일 거래대금 상위 200개만 정밀 분석(기본)
  python v2_etf_scanner.py --top-volume 100        # 더 좁게
  python v2_etf_scanner.py --refresh-meta          # 전체 ETF 당일 시세 다시 조회(보통 캐시 재사용)
  python v2_etf_scanner.py --liquidity-pct 0.8     # 20일평균거래대금 상위 20%만 남김(기본 0.7=상위 30%)

전체 ETF(~1,100여개)의 당일 시세(inquire_price)는 한 번 받아서 캐시에 저장해두고
(state/etf_meta_cache.json) 재사용한다 — 그래야 --top-volume·--liquidity-pct 값을 바꿔가며
여러 번 돌려도 매번 1,100번씩 API를 다시 부르지 않는다. 캐시가 오래됐으면 --refresh-meta.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from kis_client import RateLimitedCaller, fetch_daily_bars, to_float
from v2_config import KST, ROOT, state_dir
from v2_kospi_master import load_etf_universe

log = logging.getLogger("v2_etf_scanner")

META_CACHE_PATH = state_dir() / "etf_meta_cache.json"

INQUIRE_PRICE_PATH = "/uapi/etfetn/v1/quotations/inquire-price"
INQUIRE_PRICE_TR = "FHPST02400000"
COMPONENT_PATH = "/uapi/etfetn/v1/quotations/inquire-component-stock-price"
COMPONENT_TR = "FHKST121600C0"
COMPONENT_SCR_CODE = "11216"

# 전용 플래그가 없어 이름 패턴으로만 배제 가능(설계노트 3절). 인버스/레버리지는 배수 필터가
# 우선 걸러내지만, 상품명에도 남아있어 이중으로 걸러둔다.
NAME_EXCLUDE = re.compile(
    r"커버드콜|프리미엄|위클리|타겟데이트|TDF|채권|합성|인버스|레버리지|액티브", re.IGNORECASE)
# 시장 전체 지수 추종 ETF는 '대표 업종'에 지수명 그대로 찍혀 나와 자동 배제가 애매함(설계노트
# "남는 수동 작업") — 라이브 스캔(--limit 60, 2026-09-23)에서 실제로 걸러지지 않고 나온 라벨들을
# 반영해뒀다. 실행 후 basket_watchlist.scanned.json 을 열어보고 새로 걸리는 라벨이 있으면 추가할 것.
SECTOR_BLACKLIST_DEFAULT = {
    "코스피", "코스피200", "코스피200 TR", "코스닥", "코스닥150", "코스피100", "코스피50",
    "KOSPI", "KOSPI200", "KOSDAQ", "KOSDAQ150", "KSQ150",
}

OUT_PATH = ROOT / "basket_watchlist.scanned.json"


def _sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "", name).strip()


def fetch_etf_meta(caller: RateLimitedCaller, code: str) -> dict:
    body = caller.get(INQUIRE_PRICE_PATH, INQUIRE_PRICE_TR, {
        "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
    })
    return body.get("output") or {}


def build_meta_table(universe: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    caller = RateLimitedCaller()
    rows = []
    codes = universe if limit is None else universe.head(limit)
    for i, row in enumerate(codes.itertuples(index=False)):
        try:
            out = fetch_etf_meta(caller, row.code)
        except Exception as e:
            log.warning("%s(%s) 메타 조회 실패: %s", row.name, row.code, e)
            continue
        if not out:
            continue
        vol, price = to_float(out.get("acml_vol")), to_float(out.get("stck_prpr"))
        rows.append({
            "code": row.code, "name": row.name,
            "mltp": to_float(out.get("etf_trc_ert_mltp")),
            "aum": to_float(out.get("etf_ntas_ttam")),
            "sector": (out.get("etf_rprs_bstp_kor_isnm") or "").strip(),
            "div_name": (out.get("etf_div_name") or "").strip(),
            # 당일 거래대금(대략치) — top-volume 1차 필터용. 정밀 유동성 판단은 add_liquidity()의
            # 20일 평균을 따로 쓴다(하루치는 당일 이벤트로 노이즈가 큼, 설계노트 3절).
            "value_now": (vol * price) if (vol is not None and price is not None) else None,
        })
        if (i + 1) % 50 == 0:
            log.info("메타 조회 %d/%d", i + 1, len(codes))
    return pd.DataFrame(rows)


def load_or_build_meta_table(universe: pd.DataFrame, limit: int | None, refresh: bool) -> pd.DataFrame:
    if not refresh and limit is None and META_CACHE_PATH.exists():
        log.info("캐시된 메타 재사용: %s (재조회하려면 --refresh-meta)", META_CACHE_PATH)
        return pd.read_json(META_CACHE_PATH, orient="records")
    meta = build_meta_table(universe, limit=limit)
    if limit is None:
        META_CACHE_PATH.write_text(meta.to_json(orient="records", force_ascii=False), encoding="utf-8")
        log.info("메타 캐시 저장: %s (%d개)", META_CACHE_PATH, len(meta))
    return meta


def top_by_volume(meta: pd.DataFrame, n: int) -> pd.DataFrame:
    valid = meta.dropna(subset=["value_now"])
    top = valid.sort_values("value_now", ascending=False).head(n)
    log.info("당일 거래대금 상위 %d개로 축소: %d → %d", n, len(meta), len(top))
    return top


def add_liquidity(meta: pd.DataFrame, days: int = 40) -> pd.DataFrame:
    """살아남은 후보만 일봉을 받아 최근 20영업일 평균 거래대금(종가×거래량)을 붙인다."""
    caller = RateLimitedCaller()
    today = datetime.now(KST).date()
    start = today - timedelta(days=days)
    values = []
    for row in meta.itertuples(index=False):
        try:
            bars = fetch_daily_bars(caller, row.code, start, today)
        except Exception as e:
            log.warning("%s(%s) 일봉 조회 실패: %s", row.name, row.code, e)
            values.append(None)
            continue
        if not bars:
            values.append(None)
            continue
        recent = bars[-20:]
        values.append(float(np.mean([b["close"] * b["volume"] for b in recent])))
    meta = meta.copy()
    meta["trading_value"] = values
    return meta


def filter_candidates(meta: pd.DataFrame, sector_blacklist: set[str]) -> pd.DataFrame:
    n0 = len(meta)
    meta = meta[meta["mltp"] == 1.0]
    log.info("배수(mltp==1) 필터: %d → %d", n0, len(meta))
    meta = meta[~meta["name"].str.contains(NAME_EXCLUDE, regex=True, na=False)]
    log.info("이름 패턴 필터 후: %d", len(meta))
    meta = meta[meta["sector"].astype(bool) & ~meta["sector"].isin(sector_blacklist)]
    log.info("섹터 라벨/블랙리스트 필터 후: %d", len(meta))
    return meta


def pick_representatives(meta: pd.DataFrame, liquidity_pct: float) -> pd.DataFrame:
    valid = meta.dropna(subset=["trading_value", "aum"])
    if valid.empty:
        return valid
    cutoff = valid["trading_value"].quantile(liquidity_pct)
    valid = valid[valid["trading_value"] >= cutoff]
    log.info("유동성(상위 %.0f%%, 20일평균거래대금≥%.0f) 필터 후: %d", (1 - liquidity_pct) * 100,
             cutoff, len(valid))
    reps = valid.sort_values("aum", ascending=False).groupby("sector", as_index=False).first()
    return reps


def fetch_top_holdings(caller: RateLimitedCaller, code: str, top_n: int) -> list[dict]:
    body = caller.get(COMPONENT_PATH, COMPONENT_TR, {
        "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
        "FID_COND_SCR_DIV_CODE": COMPONENT_SCR_CODE,
    })
    output2 = body.get("output2") or []
    rows = []
    for r in output2:
        c = str(r.get("stck_shrn_iscd") or "").strip()
        if not re.fullmatch(r"\d{6}", c):
            continue
        w = to_float(r.get("etf_cnfg_issu_rlim"))
        if w is None:
            w = to_float(r.get("etf_vltn_amt"))
        if w is None:
            continue
        rows.append({"code": c, "name": (r.get("hts_kor_isnm") or "").strip(), "w": w})
    rows.sort(key=lambda x: x["w"], reverse=True)
    return rows[:top_n]


def build_basket_watchlist(representatives: pd.DataFrame, top_n: int) -> dict:
    caller = RateLimitedCaller()
    stocks, themes = [], {}
    for rep in representatives.itertuples(index=False):
        holdings = fetch_top_holdings(caller, rep.code, top_n)
        if not holdings:
            log.warning("%s(%s, %s) 구성종목 없음 — 건너뜀", rep.name, rep.code, rep.sector)
            continue
        for h in holdings:
            stocks.append({"code": h["code"], "name": h["name"], "theme": rep.sector})
        basket_name = f"{_sanitize(rep.sector)}_scan"
        themes[rep.sector] = {
            "basket_name": basket_name,
            "target": {"code": rep.code, "name": rep.name},
        }
        log.info("대표 ETF: %s(%s) 섹터=%s AUM=%.0f 구성종목 %d개",
                  rep.name, rep.code, rep.sector, rep.aum, len(holdings))
    return {
        "_설명": "v2_etf_scanner.py 가 자동 생성한 후보 목록입니다. basket_watchlist.json 을 "
                 "덮어쓰지 않습니다 — 검토 후 필요한 테마만 골라 basket_watchlist.json 에 직접 "
                 "옮겨 넣으세요.",
        "stocks": stocks,
        "themes": themes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="ETF 유니버스 앞 N개만 스캔(동작 확인용, 캐시 안 씀)")
    ap.add_argument("--top-volume", type=int, default=200,
                     help="당일 거래대금 상위 N개만 정밀 분석 대상으로 남김(기본 200)")
    ap.add_argument("--top-n", type=int, default=10, help="섹터 대표 ETF당 담을 구성종목 수")
    ap.add_argument("--liquidity-pct", type=float, default=0.7,
                     help="20일평균거래대금 백분위 컷(기본 0.7=상위 30%%만 남김)")
    ap.add_argument("--refresh-master", action="store_true", help="종목마스터 재다운로드")
    ap.add_argument("--refresh-meta", action="store_true", help="전체 ETF 당일 시세 캐시 무시하고 재조회")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    universe = load_etf_universe(force=args.refresh_master)
    log.info("ETF 유니버스 %d개 로드", len(universe))

    meta = load_or_build_meta_table(universe, limit=args.limit, refresh=args.refresh_meta)
    meta = top_by_volume(meta, args.top_volume)
    meta = filter_candidates(meta, SECTOR_BLACKLIST_DEFAULT)
    if meta.empty:
        log.error("1차 필터 통과 종목 없음 — 블랙리스트/이름 패턴을 확인하세요")
        return 1

    meta = add_liquidity(meta)
    reps = pick_representatives(meta, args.liquidity_pct)
    if reps.empty:
        log.error("유동성 필터 통과 종목 없음 — --liquidity-pct 를 낮춰보세요")
        return 1
    log.info("섹터 대표 ETF %d개 확정: %s", len(reps), ", ".join(reps["sector"]))

    payload = build_basket_watchlist(reps, args.top_n)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("저장 완료: %s (테마 %d개, 종목 %d개) — 검토 후 basket_watchlist.json 에 반영하세요",
              OUT_PATH, len(payload["themes"]), len(payload["stocks"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
