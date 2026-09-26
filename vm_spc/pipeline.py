"""VM-SPC Core 챔피언-챌린저 EOD 파이프라인 (명세 2.9) → results/{basket}/vm_spc/vm_spc_dashboard_data.json
→ dashboard_v2.html 의 "챔피언-챌린저" 탭.

Human-in-the-Loop: 주문 실행 없음. 장 마감 후 1회 돌려 익일 시나리오 라인을 만들고 판단은 사람이 한다.
Stage1(v2_data_collector)이 저장한 data/{basket}/*.csv 를 그대로 읽는다 — API 를 호출하지 않는다.

기본 라벨링은 상대순위다(2026-09-26 재보정). 절대방향(labeling.attach_labels)은 데드존 때문에
표본이 적어 신뢰구간이 넓고(17개 바스켓 중 1개만 CI 하한>50%), 상대순위는 표본이 늘어 CI가 4배
좁아지며 16/17 바스켓에서 edge 존재 자체는 통계적으로 확인됐다 — 대신 Gate 1 임계값도 그 실측
수준(0.60, gate_decision.py 참고)으로 낮췄고 통과해도 "고신뢰"가 아니라 "약한 방향성 참고 신호"다.

  python vm_spc/pipeline.py                                # basket_watchlist.json 전체 바스켓(상대순위)
  python vm_spc/pipeline.py --basket semiconductor_to_etf
  python vm_spc/pipeline.py --no-render                    # JSON 만 (대시보드 재렌더링 생략)
  python vm_spc/pipeline.py --label-mode absolute           # 비교용 — 절대방향 라벨링(구버전 방식)
                                                             # → vm_spc/vm_spc_dashboard_data_absolute.json
                                                             # (대시보드 챔피언-챌린저 탭은 상대순위 결과만 읽는다)
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
from pathlib import Path

import numpy as np
import pandas as pd

from v2_compute_engine import build_sensor_features
from v2_config import KST, Params, get_basket, load_baskets, results_dir
from v2_datastore import load_bars
from v2_signals import vm_scores
from vm_spc import evaluation, gate_decision, labeling, splitter, universe
from vm_spc.features import FEATURE_COLS, FEATURE_LABELS, build_feature_frame
from vm_spc.models.legacy_rule import LEGACY_THRESHOLD, predict_legacy
from vm_spc.models.lgbm_challenger import LGBM_PARAMS, fit_lgbm, get_gain_importance, predict_proba_lgbm
from vm_spc.models.ridge_baseline import fit_ridge, get_standardized_coefs, predict_proba_ridge

log = logging.getLogger("vm_spc.pipeline")

MIN_EFFECTIVE_N = 300     # 명세 1.5: 총 유효표본 N ≥ 300
MODELS = ["legacy", "baseline", "challenger"]


def vm_spc_results_dir(basket_name: str) -> Path:
    d = results_dir() / basket_name / "vm_spc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vm_spc_data_path(basket_name: str, label_mode: str = "relative") -> Path:
    """상대순위(기본, 2026-09-26부터 정식 채택)는 기존 파일명을 그대로 써서 대시보드가 읽고,
    다른 라벨링 모드는 별도 파일에 저장해 기본 결과를 덮어쓰지 않는다(대시보드는 상대순위 파일만 읽는다)."""
    suffix = "" if label_mode == "relative" else f"_{label_mode}"
    return vm_spc_results_dir(basket_name) / f"vm_spc_dashboard_data{suffix}.json"


def _design_matrix(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """5대 피처 + Ticker ID 더미(통제변수). 더미 열 집합은 유니버스 기준으로 고정해 fold 마다 같게."""
    X = df[FEATURE_COLS].astype(float).reset_index(drop=True)
    tk = pd.Categorical(df["ticker_id"].to_numpy(), categories=tickers)
    dummies = pd.get_dummies(tk, prefix="tk", dtype=float)
    dummies.columns = [str(c) for c in dummies.columns]
    return pd.concat([X, dummies.reset_index(drop=True)], axis=1)


def _vm_score_panel(basket: dict, p: Params) -> pd.DataFrame:
    """Legacy 룰 입력: 종목별 VM Score (기존 Layer 1 스코어카드, vm_predict 와 동일 계산)."""
    feats, rs = build_sensor_features(basket, p)
    vm = vm_scores(feats, rs, p)
    rows = [pd.DataFrame({"date": df.index, "ticker_id": code, "vm_score": df["vm_score"].to_numpy()})
            for code, df in vm.items()]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "ticker_id", "vm_score"])


def _r(v, n=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)


def _round_metrics(m: dict) -> dict:
    return {k: (_r(v) if isinstance(v, float) else v) for k, v in m.items()}


def run_eod_pipeline(basket: dict, p: Params | None = None, label_mode: str = "relative") -> dict | None:
    p = p or Params()
    if label_mode not in labeling.LABEL_MODES:
        raise ValueError(f"알 수 없는 라벨링 모드: {label_mode!r} (가능한 값: {labeling.LABEL_MODES})")
    name_by_code = {s["code"]: s["name"] for s in basket["sensors"]}
    as_of = datetime.now(KST).strftime("%Y-%m-%d")
    tickers = universe.get_point_in_time_constituents(as_of, basket)

    target_bars = load_bars(basket["name"], basket["target"]["code"])
    if target_bars is None or target_bars.empty:
        log.error("타겟 %s 일봉 없음 — v2_data_collector.py 먼저 실행", basket["target"]["name"])
        return None
    ticker_bars = {c: b for c in tickers if (b := load_bars(basket["name"], c)) is not None and not b.empty}
    tickers = [c for c in tickers if c in ticker_bars]
    if len(tickers) < 2:
        log.error("유효 종목 %d개 — 풀링 패널 구성 불가", len(tickers))
        return None

    log.info("Layer 2 피처 산출 중 (%d종목)...", len(tickers))
    features = build_feature_frame(ticker_bars, target_bars, p)
    features = features.merge(_vm_score_panel(basket, p), on=["date", "ticker_id"], how="left")
    features = features[universe.membership_mask(features["date"], features["ticker_id"], basket)]
    complete = features.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    n_before = int(complete.groupby("ticker_id")["close"].shift(-1).notna().sum())
    label_kwargs = {}
    if label_mode == "relative":
        # MIN_ACTIVE(기본 6)가 바스켓 종목수보다 크거나 같으면 그날 활성 종목이 하루도 6개를
        # 못 채워 전체가 통째로 제외된다(5종목 바스켓에서 실제로 재현됨) — 종목수보다 하나
        # 적은 값으로 캡을 씌워 항상 순위가 가능하게 한다. 7종목 이상 바스켓은 그대로 6이라
        # 기존 검증치와 동일하다.
        label_kwargs["min_active"] = min(labeling.MIN_ACTIVE, max(2, len(tickers) - 1))
    labeled = labeling.build_labels(complete, mode=label_mode, **label_kwargs)
    n_excluded = n_before - len(labeled)

    X_all = _design_matrix(labeled, tickers)
    y_all = labeled["y"].to_numpy()
    splits = splitter.walk_forward_splits(labeled["date"])
    if not splits:
        # 이전 실행 결과가 남아 있으면 대시보드가 낡은 판정을 계속 보여준다 — 지금 상태(검증 불가)에 맞게 지운다.
        vm_spc_data_path(basket["name"], label_mode).unlink(missing_ok=True)
        log.error("Walk-Forward fold 를 하나도 못 만듦 — 유효 거래일 %d일 (최소 %d일 필요)",
                  labeled["date"].nunique(), splitter.TRAIN_DAYS + splitter.EMBARGO_DAYS + 20)
        return None

    log.info("Walk-Forward %d folds 검증 중...", len(splits))
    oos_parts, fold_rows = [], []
    for k, (tr, te, meta) in enumerate(splits):
        ridge = fit_ridge(X_all.iloc[tr], y_all[tr])
        lgbm = fit_lgbm(X_all.iloc[tr], y_all[tr])
        part = labeled.iloc[te][["date", "ticker_id", "y", "next_ret", "vm_score"]].copy()
        part["fold"] = k + 1
        part["p_legacy"] = predict_legacy(part["vm_score"])
        part["p_baseline"] = predict_proba_ridge(ridge, X_all.iloc[te])
        part["p_challenger"] = predict_proba_lgbm(lgbm, X_all.iloc[te])
        oos_parts.append(part)
        row = {"fold": k + 1, **meta, "n_train": int(len(tr)), "n_test": int(len(te))}
        for m in MODELS:
            fm = evaluation.compute_metrics(part["y"], part[f"p_{m}"], part["date"], with_ci=False)
            row[m] = {"high_conf_precision": _r(fm["high_conf_precision"]), "n_s_core": fm["n_s_core"],
                      "precision": _r(fm["precision"]), "n_s_base": fm["n_s_base"]}
        fold_rows.append(row)
    oos = pd.concat(oos_parts, ignore_index=True)

    log.info("OOS 풀링 지표 + 날짜 클러스터 부트스트랩(B=%d) 계산 중...", evaluation.N_BOOT)
    agg = evaluation.aggregate_fold_metrics(oos, MODELS)
    agg["legacy"]["brier"] = None   # 0/1 이진 판정이라 Brier 는 오분류율과 같아져 의미 없음
    decision = gate_decision.decide_champion(agg["baseline"], agg["challenger"])
    log.info("판정: %s — %s", decision["champion"], decision["reason"])

    # ── 운영 단계: 최근 TRAIN_DAYS 거래일로 재학습 → 최신 EOD 피처로 익일 시나리오 ──
    uniq = np.array(sorted(labeled["date"].unique()))
    recent = labeled["date"] >= uniq[max(0, len(uniq) - splitter.TRAIN_DAYS)]
    ridge_f = fit_ridge(X_all[recent.to_numpy()], y_all[recent.to_numpy()])
    lgbm_f = fit_lgbm(X_all[recent.to_numpy()], y_all[recent.to_numpy()])
    latest_date = complete["date"].max()
    today = complete[complete["date"] == latest_date].reset_index(drop=True)
    X_today = _design_matrix(today, tickers)
    p_base_t = predict_proba_ridge(ridge_f, X_today)
    p_chal_t = predict_proba_lgbm(lgbm_f, X_today)
    champion = decision["champion"]
    scenario = []
    for i, r in today.iterrows():
        p_champ = {"BASELINE": p_base_t[i], "CHALLENGER": p_chal_t[i]}.get(champion)
        if p_champ is None:
            tier = None
        elif p_champ >= evaluation.TIER2_THRESHOLD:
            tier = "S_core"
        elif p_champ > evaluation.TIER1_THRESHOLD:
            tier = "S_base"
        else:
            tier = "none"
        close = float(r["close"])
        # line_up/down 은 "종가 대비 절대 등락률 기준선"이라 절대방향 모드에서만 의미가 있다 —
        # 상대순위 모드는 그날 바스켓 내 상대적 우열이 기준이라 고정된 가격선이 없다.
        line_up = round(close * (1 + labeling.DEADZONE_UP), 2) if label_mode == "absolute" else None
        line_down = round(close * (1 + labeling.DEADZONE_DOWN), 2) if label_mode == "absolute" else None
        scenario.append({
            "code": r["ticker_id"], "name": name_by_code.get(r["ticker_id"], r["ticker_id"]),
            "close": close,
            "line_up": line_up,
            "line_down": line_down,
            "vm_score": _r(r["vm_score"]),
            "legacy_signal": bool(predict_legacy([r["vm_score"]])[0]),
            "p_baseline": _r(p_base_t[i]), "p_challenger": _r(p_chal_t[i]),
            "p_champion": _r(p_champ), "tier": tier,
            "features": {f: _r(r[f]) for f in FEATURE_COLS},
        })
    scenario.sort(key=lambda s: -(s["p_champion"] if s["p_champion"] is not None else s["vm_score"] or 0))

    limitations = []
    if not universe.is_point_in_time():
        limitations.append(universe.SURVIVORSHIP_WARNING)
    if agg["baseline"]["n_eval"] < MIN_EFFECTIVE_N:
        limitations.append(f"OOS 유효표본 {agg['baseline']['n_eval']}건 — 명세 기준 N ≥ {MIN_EFFECTIVE_N} 미달.")
    for m in ("baseline", "challenger"):
        if agg[m]["n_s_core"] < 30:
            limitations.append(f"{m} S_core 표본 {agg[m]['n_s_core']}건 — High-Conf Precision 추정이 불안정합니다.")
    limitations.append("coint_z 는 ETF-ex-A 대신 타겟 ETF 자체 대비 롤링 OLS 스프레드입니다(전 종목 편입비중 미확보).")

    payload = {
        "basket": basket["name"],
        "generated_at": datetime.now(KST).isoformat(),
        "as_of_feature_date": latest_date.strftime("%Y-%m-%d"),
        "universe": {"tickers": [{"code": c, "name": name_by_code.get(c, c)} for c in tickers],
                     "point_in_time": universe.is_point_in_time()},
        "feature_labels": FEATURE_LABELS,
        "config": {
            "label_mode": label_mode,
            "deadzone": [labeling.DEADZONE_DOWN, labeling.DEADZONE_UP] if label_mode == "absolute" else None,
            "relative_rank": ({"top_frac": labeling.TOP_FRAC, "bot_frac": labeling.BOT_FRAC,
                               "min_active": labeling.MIN_ACTIVE} if label_mode == "relative" else None),
            "train_days": splitter.TRAIN_DAYS, "test_days": splitter.TEST_DAYS,
            "embargo_days": splitter.EMBARGO_DAYS, "purge_days": splitter.PURGE_DAYS,
            "tier1": evaluation.TIER1_THRESHOLD, "tier2": evaluation.TIER2_THRESHOLD,
            "gate1": gate_decision.GATE1_THRESHOLD, "gate1_ci_lower_min": gate_decision.CI_LOWER_MIN,
            "gate1_min_n_core": gate_decision.MIN_N_CORE, "gate2_margin": gate_decision.GATE2_MARGIN,
            "n_boot": evaluation.N_BOOT, "legacy_threshold": LEGACY_THRESHOLD, "lgbm": LGBM_PARAMS,
        },
        "sample": {"n_labeled": int(len(labeled)), "n_excluded": int(n_excluded),
                   "n_oos": int(len(oos)), "n_oos_days": int(oos["date"].nunique()),
                   "oos_start": oos["date"].min().strftime("%Y-%m-%d"),
                   "oos_end": oos["date"].max().strftime("%Y-%m-%d")},
        "metrics": {m: _round_metrics(agg[m]) for m in MODELS},
        "folds": fold_rows,
        "decision": {**decision, "gate2": ({k: (_r(v) if isinstance(v, float) else v)
                                            for k, v in decision["gate2"].items()} if decision["gate2"] else None)},
        "contributions": {
            "ridge_std_coef": {k: _r(v) for k, v in get_standardized_coefs(ridge_f, FEATURE_COLS).items()},
            "lgbm_gain_share": {k: _r(v) for k, v in get_gain_importance(lgbm_f, FEATURE_COLS).items()},
            "train_window": [pd.Timestamp(uniq[max(0, len(uniq) - splitter.TRAIN_DAYS)]).strftime("%Y-%m-%d"),
                             pd.Timestamp(uniq[-1]).strftime("%Y-%m-%d")],
        },
        "scenario": scenario,
        "limitations": limitations,
    }
    out_path = vm_spc_data_path(basket["name"], label_mode)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("저장 완료: %s", out_path)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket", help="생략 시 basket_watchlist.json 의 모든 바스켓")
    ap.add_argument("--no-render", action="store_true", help="대시보드 HTML 재렌더링 생략")
    ap.add_argument("--label-mode", choices=list(labeling.LABEL_MODES), default="relative",
                    help="relative(기본, 2026-09-26부터 정식 채택 — 그날 바스켓 내 상대순위, labeling.py 참고) / "
                         "absolute(비교용, 데드존 절대방향). absolute 는 별도 파일"
                         "(vm_spc_dashboard_data_absolute.json)에만 저장되고 "
                         "대시보드 챔피언-챌린저 탭은 계속 relative 결과를 보여준다.")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    baskets = [get_basket(args.basket)] if args.basket else load_baskets()
    failed = []
    for basket in baskets:
        log.info("=== VM-SPC Core '%s' (label_mode=%s) ===", basket["name"], args.label_mode)
        try:
            if run_eod_pipeline(basket, label_mode=args.label_mode) is None:
                failed.append(basket["name"])
                continue
        except Exception:
            log.exception("바스켓 '%s' 실패 — 건너뜀", basket["name"])
            failed.append(basket["name"])
            continue
        if not args.no_render and args.label_mode == "relative":
            # 대시보드는 상대순위 파일만 읽으므로, absolute 모드에서는 재렌더링해도 반영되지 않는다.
            from v2_render_dashboard import render_basket
            render_basket(basket)
    if failed:
        log.warning("실패/생략 바스켓: %s", failed)
    return 1 if failed and len(failed) == len(baskets) else 0


if __name__ == "__main__":
    sys.exit(main())
