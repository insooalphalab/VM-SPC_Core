"""Stage 1+2: 대표종목 A vs ETF-ex-A 공적분 회귀 + SPC 잔차 연산 (스펙 4절).

Stage 0(pair_representativeness.py)이 확정한 대표종목이 있는 바스켓만 대상으로 한다 — 대표종목이
없으면(selected_representative=null) 이 단계는 실행하지 않는다(스펙 2.4 "이 파일이 있어야만
다음 단계가 실행됨").

  python pair_cointegration.py --basket <바스켓명>
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

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from pair_config import PairParams, pair_results_dir
from v2_config import get_basket
from v2_datastore import load_bars

log = logging.getLogger("pair_cointegration")


def build_exself_index(target_bars: pd.DataFrame, a_bars: pd.DataFrame, w_a: float) -> pd.Series:
    """ETF-ex-A 수익률을 누적해 만든 지수(기준 100) — pair_representativeness.py와 동일한 정의."""
    r_etf = target_bars["close"].pct_change()
    r_a = a_bars["close"].pct_change()
    idx = r_etf.index.intersection(r_a.index)
    r_exa = (r_etf.reindex(idx) - w_a * r_a.reindex(idx)) / (1 - w_a)
    r_exa = r_exa.dropna()
    return 100.0 * (1 + r_exa).cumprod()


def half_life(spread: pd.Series) -> float | None:
    """OU 반감기: Δspread(t) = θ·spread(t-1) + ε 회귀의 θ<0(평균회귀)일 때만 정의된다."""
    s = spread.dropna()
    lag, delta = s.shift(1).dropna(), s.diff().dropna()
    idx = lag.index.intersection(delta.index)
    lag, delta = lag.reindex(idx), delta.reindex(idx)
    if len(idx) < 30:
        return None
    x = sm.add_constant(lag.to_numpy())
    model = sm.OLS(delta.to_numpy(), x).fit()
    theta = model.params[1]
    if theta >= 0:
        return None  # 평균회귀 성질 없음 — 반감기 정의 불가
    return float(-np.log(2) / np.log(1 + theta))


def confirm_states(z: pd.Series, threshold: float, hysteresis_days: int) -> pd.Series:
    """연속 hysteresis_days 거래일 동안 같은 방향(정상 복귀 포함)이 유지돼야 상태를 확정한다
    (플리커링 방지, 스펙 5절 Step4). 상태값: 'normal' | 'alert_up' | 'alert_down'.

    확정 상태(current)와 '전환 후보(candidate)'를 분리한다 — candidate가 hysteresis_days 연속
    관측되어야 current로 승격되고, 그 전까지는 이전 확정 상태가 그대로 유지된다. 이 규칙은
    이탈 진입뿐 아니라 정상 복귀에도 동일하게 적용된다(복귀도 노이즈 한 번으로 바로 풀리지 않음)."""
    raw = pd.Series("normal", index=z.index, dtype=object)
    raw[z >= threshold] = "alert_up"
    raw[z <= -threshold] = "alert_down"

    confirmed = pd.Series("normal", index=z.index, dtype=object)
    current = "normal"
    candidate, candidate_run = None, 0
    for i in range(len(raw)):
        if pd.isna(z.iloc[i]):
            candidate, candidate_run = None, 0
            confirmed.iloc[i] = current
            continue
        val = raw.iloc[i]
        if val == current:
            candidate, candidate_run = None, 0
        else:
            if val == candidate:
                candidate_run += 1
            else:
                candidate, candidate_run = val, 1
            if candidate_run >= hysteresis_days:
                current = candidate
                candidate, candidate_run = None, 0
        confirmed.iloc[i] = current
    return confirmed


def analyze_pair(basket_name: str, etf_code: str, target_bars: pd.DataFrame, rep_code: str,
                  a_bars: pd.DataFrame, rep: dict, p: PairParams) -> dict:
    """대표종목 A(rep_code) vs ETF-ex-A 공적분·SPC 분석 본체. run()과 종목비교(candidate 강제
    지정) 양쪽에서 재사용한다 — 어떤 종목을 'A'로 볼지만 바꿔 넣으면 나머지 계산은 동일하다."""
    exself_index = build_exself_index(target_bars, a_bars, rep["weight"])
    log_a = np.log(a_bars["close"])
    idx = log_a.index.intersection(exself_index.index)
    log_a, log_b = log_a.reindex(idx), np.log(exself_index.reindex(idx))

    x = sm.add_constant(log_b.to_numpy())
    model = sm.OLS(log_a.to_numpy(), x).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    spread = pd.Series(log_a.to_numpy() - (alpha + beta * log_b.to_numpy()), index=idx)

    adf_stat, adf_p, *_ = adfuller(spread.dropna(), result_object=False)
    hl = half_life(spread)

    roll_mean = spread.rolling(p.spc_window, min_periods=p.spc_window // 2).mean().shift(1)
    roll_std = spread.rolling(p.spc_window, min_periods=p.spc_window // 2).std().shift(1)
    z = (spread - roll_mean) / roll_std
    states = confirm_states(z, p.spc_z_threshold, p.hysteresis_days)

    current_state = states.dropna().iloc[-1] if len(states.dropna()) else "normal"
    alarm_dates = states.index[(states != "normal") & (states.shift(1) == "normal")]

    return {
        "basket": basket_name,
        "etf_code": etf_code,
        "representative": {"code": rep_code, "name": rep["name"], "weight": rep["weight"],
                            "exself_corr": rep["exself_corr"], "leadlag_corr": rep["leadlag_corr"]},
        "cointegration": {"alpha": alpha, "beta": beta, "adf_stat": float(adf_stat), "adf_p": float(adf_p),
                           "is_cointegrated": bool(adf_p < p.adf_alpha), "half_life_days": hl},
        "params": {"spc_window": p.spc_window, "spc_z_threshold": p.spc_z_threshold,
                   "hysteresis_days": p.hysteresis_days},
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in idx],
            "price_a": [None if pd.isna(v) else round(float(v), 2) for v in a_bars["close"].reindex(idx)],
            "index_exa": [None if pd.isna(v) else round(float(v), 4) for v in exself_index.reindex(idx)],
            "spread": [None if pd.isna(v) else round(float(v), 6) for v in spread],
            "z": [None if pd.isna(v) else round(float(v), 4) for v in z],
            "state": [str(s) for s in states],
        },
        "current_state": current_state,
        "alarm_dates": [d.strftime("%Y-%m-%d") for d in alarm_dates],
    }


def run(basket: dict, p: PairParams, force_code: str | None = None) -> dict | None:
    stage0_path = pair_results_dir(basket["name"]) / "stage0_representativeness.json"
    if not stage0_path.exists():
        log.error("stage0_representativeness.json 없음 — pair_representativeness.py 먼저 실행")
        return None
    stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
    rep_code = force_code or stage0.get("selected_representative")
    if not rep_code:
        log.info("바스켓 '%s': Stage0에서 대표종목 없음 — Stage1/2 생략(정직한 결론: 단일 종목 대변 불가)",
                  basket["name"])
        stale = pair_results_dir(basket["name"]) / "pair_analysis_result.json"
        if stale.exists():
            # 이전 실행(그때는 대표종목이 있었을 때) 결과 파일이 남아있으면 대시보드가 낡은
            # 선정 결과를 계속 보여주게 된다 — 지금 판정(대표종목 없음)과 맞게 지운다.
            stale.unlink()
            log.info("낡은 pair_analysis_result.json 삭제")
        return None
    rep = next((c for c in stage0["candidates"] if c["code"] == rep_code), None)
    if rep is None:
        log.error("종목코드 %s는 Stage0 후보 목록에 없음 — pair_representativeness.py의 후보 수(n_candidates)를 늘려야 할 수 있음", rep_code)
        return None

    target_bars = load_bars(basket["name"], stage0["etf_code"])
    a_bars = load_bars(basket["name"], rep_code)
    if target_bars is None or a_bars is None:
        log.error("일봉 데이터 없음 — pair_representativeness.py를 먼저 실행했는지 확인")
        return None

    payload = analyze_pair(basket["name"], stage0["etf_code"], target_bars, rep_code, a_bars, rep, p)
    payload["etf_name"] = stage0["etf_name"]

    out_path = pair_results_dir(basket["name"]) / "pair_analysis_result.json"
    if force_code:
        log.warning("--force-code로 강제 지정된 결과입니다. 대시보드에 반영하려면 이대로 저장하되,"
                    " Stage0의 자동선정 결과와 다르다는 점을 알고 있어야 합니다.")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("바스켓 '%s' (대표종목=%s): ADF p=%.4f(공적분=%s) half-life=%s 현재상태=%s",
              basket["name"], rep["name"], payload["cointegration"]["adf_p"],
              payload["cointegration"]["is_cointegrated"], payload["cointegration"]["half_life_days"],
              payload["current_state"])
    log.info("저장 완료: %s", out_path)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket")
    ap.add_argument("--force-code", help="Stage0 자동선정 대신 이 종목코드를 대표종목으로 강제 지정(비교용)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    basket = get_basket(args.basket)
    run(basket, PairParams(), force_code=args.force_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
