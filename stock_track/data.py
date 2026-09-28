"""기준 지수·소속 시장·ETF 내 비중. 공용 core/ 는 수정하지 않고 가져다 쓴다."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
from datetime import datetime

import pandas as pd

from kis_client import RateLimitedCaller
from v2_config import KST, results_dir
from v2_data_collector import update_bars
from v2_datastore import load_bars

BENCHMARKS = {
    "KOSPI": {"code": "069500", "name": "KODEX 200"},
    "KOSDAQ": {"code": "229200", "name": "KODEX 코스닥150"},
}
BM_BASKET = "_benchmarks"     # data/_benchmarks/{code}.csv


def kospi_codes() -> set[str]:
    """KOSPI 종목마스터에 있으면 코스피, 없으면 코스닥으로 본다(국내 바스켓 종목은 이 둘뿐)."""
    from v2_kospi_master import load_master
    return set(load_master()["단축코드"].astype(str).str.strip())


def market_of(code: str, kospi: set[str]) -> str:
    return "KOSPI" if code in kospi else "KOSDAQ"


def update_benchmarks(caller: RateLimitedCaller, days: int) -> None:
    for bm in BENCHMARKS.values():
        update_bars(caller, BM_BASKET, bm["code"], days)


def load_benchmark(market: str) -> pd.DataFrame | None:
    return load_bars(BM_BASKET, BENCHMARKS[market]["code"])


def weights_path(basket_name: str):
    return results_dir() / basket_name / "stock_track" / "weights.json"


def update_weights(caller: RateLimitedCaller, basket: dict) -> dict:
    """타겟 ETF 구성종목 비중(%) — 센서가 보유목록에 없으면 None(화면에 "미편입")."""
    from v2_etf_scanner import fetch_top_holdings
    holdings = {h["code"]: h["w"] for h in fetch_top_holdings(caller, basket["target"]["code"], 500)}
    out = {"as_of": datetime.now(KST).strftime("%Y-%m-%d"),
           "weights": {s["code"]: holdings.get(s["code"]) for s in basket["sensors"]}}
    path = weights_path(basket["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_weights(basket_name: str) -> dict:
    path = weights_path(basket_name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"as_of": None, "weights": {}}
