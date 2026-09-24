"""Stage 0(신규): 섹터 대표종목 실증 선정 (스펙 2절).

ETF 가격은 정의상 구성종목의 가중합이라, 종목 A와 ETF 전체의 상관은 A가 산식에 직접 포함돼
있다는 사실만으로 이미 하한선을 보장받는다 — "닮았다"는 결과가 나와도 실질적 대표성 때문인지
단순 포함관계 때문인지 구분이 안 된다(스펙 1절). 이를 깨려면 A 자신의 기여분을 뺀 나머지
(ETF-ex-A)와 A를 비교해야 한다:

    r_ETF-ex-A(t) = (r_ETF(t) - w_A * r_A(t)) / (1 - w_A)

스펙 원문은 가격 레벨로 이 식을 쓰지만, ETF 시장가(주당)와 구성종목 가격은 단위가 다른(ETF는
순자산가치를 좌수로 나눈 값이라 종목 가격과 직접 더하고 뺄 수 없다) 별개의 스케일이라 가격
레벨에 그대로 적용하면 왜곡된다. 그래서 여기서는 **수익률(로그 대신 산술 수익률)** 에 이 식을
적용하고, 필요하면 그 수익률을 누적해 지수화한다 — 개념(=A의 가중 기여분을 빼고 나머지를
재정규화)은 스펙과 동일하고 계산 단위만 오류가 안 나게 바꾼 것.

대표종목 선정은 2단계다(2026-09-24 개정):
  1) 동시상관(exself_corr) >= exself_corr_min 을 통과한 종목만 "대변 가능한 후보"로 남긴다.
  2) 그 후보들에 한해 실제 공적분 검정(pair_cointegration.analyze_pair, ADF p<adf_alpha)을
     돌려서, 공적분이 확인된 종목 중 ADF p가 가장 작은(=가장 강하게 공적분인) 종목을 최종 대표로
     뽑는다. 처음엔 동시상관 1등을 그대로 최종 대표로 썼는데, 4개 바스켓으로 강제 비교해보니
     4/4 전부 "동시상관 1등 ≠ 공적분 제일 잘 되는 종목"이었다(반도체: SK하이닉스 1등인데 ADF
     p=0.53, 삼성전자는 2등인데 p=0.046 — 반대로 전력기기는 LS가 5위권 동시상관인데 p=0.0004로
     압도적). 동시상관은 "대변 가능한가"를 걸러주는 필터일 뿐, 뒤에 실제로 쓸 PAIR-SPC 관리도가
     의미 있으려면 공적분이 있어야 하므로 최종 선정 기준을 공적분으로 바꿨다. 후보 전부 exself는
     통과했는데 공적분이 하나도 안 되면(코스피10에서 실제로 이런 경우가 나옴) 대표종목 없음으로
     처리한다 — 억지로 진행하지 않는다는 스펙 원칙은 그대로 유지.

  python pair_representativeness.py --basket <바스켓명>
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
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from kis_client import RateLimitedCaller, fetch_daily_bars
from pair_config import PairParams, pair_results_dir
from pair_cointegration import analyze_pair
from v2_config import KST, get_basket
from v2_datastore import load_bars, save_bars
from v2_etf_scanner import fetch_top_holdings

log = logging.getLogger("pair_representativeness")


def _ensure_bars(basket_name: str, code: str, days: int, caller: RateLimitedCaller) -> pd.DataFrame | None:
    """이미 수집된 종목이면 재사용, 아니면(대표종목 후보가 기존 센서 목록 밖일 수 있음) 새로 수집."""
    bars = load_bars(basket_name, code)
    if bars is not None and not bars.empty:
        return bars
    from datetime import timedelta
    today = datetime.now(KST).date()
    rows = fetch_daily_bars(caller, code, today - timedelta(days=days), today)
    if not rows:
        return None
    save_bars(basket_name, code, rows)
    return load_bars(basket_name, code)


def _exself_returns(r_etf: pd.Series, r_a: pd.Series, w_a: float) -> pd.Series:
    idx = r_etf.index.intersection(r_a.index)
    return (r_etf.reindex(idx) - w_a * r_a.reindex(idx)) / (1 - w_a)


def evaluate_candidate(r_etf: pd.Series, r_a: pd.Series, w_a: float, p: PairParams) -> dict:
    r_exa = _exself_returns(r_etf, r_a, w_a)
    idx = r_a.index.intersection(r_exa.index)
    a, exa = r_a.reindex(idx), r_exa.reindex(idx)
    exself_corr = float(a.corr(exa)) if len(idx) > 20 else None

    leadlag = {}
    for k in p.leadlag_days:
        # A(t) vs ex-A(t+k): A가 미래(t+k)의 나머지 섹터를 선행하는지 — exa를 k일 앞으로 당겨서(shift(-k))
        # 같은 t 인덱스에 정렬한 뒤 상관.
        shifted = exa.shift(-k)
        mask = a.notna() & shifted.notna()
        leadlag[f"k{k}"] = float(a[mask].corr(shifted[mask])) if mask.sum() > 20 else None

    # 대표성 판정은 동시상관(exself_corr) 단독 기준으로 한다. 실측 결과 리드-래그(k=1~3)는
    # 대표성과 무관하게 거의 항상 0 근처(-0.1~0.06)로 나왔다 — 일별 수익률 자체가 원래 자기상관이
    # 거의 없는(효율적시장) 성질이라, "리드-래그가 동시상관만큼 유지돼야 한다"는 원래 기준은 종목
    # 대표성과 무관하게 구조적으로 항상 불합격하는 결함이 있었다(사용자 확인 후 제거, 2026-09-24).
    # 리드-래그 값 자체는 참고정보로 계속 저장한다.
    is_representative = exself_corr is not None and exself_corr >= p.exself_corr_min

    return {"exself_corr": exself_corr, "leadlag_corr": leadlag, "is_representative": is_representative}


def run(basket: dict, p: PairParams) -> dict:
    caller = RateLimitedCaller()
    target_code = basket["target"]["code"]
    target_bars = _ensure_bars(basket["name"], target_code, p.history_days, caller)
    if target_bars is None or target_bars.empty:
        raise RuntimeError(f"타겟 {target_code} 일봉을 확보할 수 없습니다.")
    r_etf = target_bars["close"].pct_change().dropna()

    holdings = fetch_top_holdings(caller, target_code, p.n_candidates)
    if not holdings:
        raise RuntimeError(f"{target_code} 구성종목 조회 실패 — 개별주 ETF가 아니거나 API 응답 없음")

    candidates = []
    for h in holdings:
        code, name, w_pct = h["code"], h["name"], h["w"]
        w_a = w_pct / 100.0
        if not (0 < w_a < 0.95):
            log.info("%s(%s): 비중 %.2f%% 비정상 — 후보 제외", name, code, w_pct)
            continue
        bars = _ensure_bars(basket["name"], code, p.history_days, caller)
        if bars is None or bars.empty:
            log.warning("%s(%s): 일봉 확보 실패 — 후보 제외", name, code)
            continue
        r_a = bars["close"].pct_change().dropna()
        result = evaluate_candidate(r_etf, r_a, w_a, p)
        candidates.append({"code": code, "name": name, "weight": round(w_a, 4), **result})

    # 2단계: 동시상관을 통과한 후보에 한해 실제 공적분 검정을 돌린다(모듈 docstring 참고).
    for c in candidates:
        if not c["is_representative"]:
            continue
        a_bars = load_bars(basket["name"], c["code"])
        coint = analyze_pair(basket["name"], target_code, target_bars, c["code"], a_bars, c, p)["cointegration"]
        c["adf_p"] = coint["adf_p"]
        c["is_cointegrated"] = coint["is_cointegrated"]
        c["half_life_days"] = coint["half_life_days"]

    cointegrated = [c for c in candidates if c.get("is_cointegrated")]
    selected = min(cointegrated, key=lambda c: c["adf_p"])["code"] if cointegrated else None

    payload = {
        "sector": basket["name"],
        "etf_code": target_code,
        "etf_name": basket["target"]["name"],
        "evaluated_date": datetime.now(KST).strftime("%Y%m%d"),
        "params": {"exself_corr_min": p.exself_corr_min, "leadlag_retain_ratio": p.leadlag_retain_ratio,
                   "leadlag_days": list(p.leadlag_days), "adf_alpha": p.adf_alpha},
        "candidates": candidates,
        "selected_representative": selected,
    }
    out_path = pair_results_dir(basket["name"]) / "stage0_representativeness.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if selected:
        name = next(c["name"] for c in candidates if c["code"] == selected)
        adf_p = next(c["adf_p"] for c in candidates if c["code"] == selected)
        log.info("바스켓 '%s': 대표종목 확정 — %s(%s), ADF p=%.4f", basket["name"], name, selected, adf_p)
    elif any(c["is_representative"] for c in candidates):
        log.info("바스켓 '%s': 동시상관 통과 종목은 있으나 공적분 통과 종목 없음 — 대표종목 없음",
                  basket["name"])
    else:
        log.info("바스켓 '%s': 대표성 기준(r>=%.2f) 통과 종목 없음 — 단일 종목으로 대변 불가",
                  basket["name"], p.exself_corr_min)
    log.info("저장 완료: %s", out_path)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    basket = get_basket(args.basket)
    run(basket, PairParams())
    return 0


if __name__ == "__main__":
    sys.exit(main())
